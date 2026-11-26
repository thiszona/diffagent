package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/scanner"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/packages"
)

const (
	// maxDirs bounds the changed directories loaded; a file past them is
	// skipped as "package cap".
	maxDirs = 8
	// maxErrors bounds the errors reported, so a broken repo's type errors can't
	// push the packs past the client's stdout cap.
	maxErrors = 20
)

// maxParseFiles caps the files one load parses: where export data is missing,
// x/tools type-checks dependencies from source, which can mean the whole module
// graph. A variable so a test can lower it.
var maxParseFiles int64 = 2000

// job is one request's state.
type job struct {
	root string // as requested
	real string // root after EvalSymlinks
	// module is the root go.mod's module path: under -trimpath, export data
	// names a main-module file by its import path, not its directory.
	module string
	src    *sources
	rels   map[string]relPath
	errs   []string
}

type relPath struct {
	path string
	ok   bool
}

// file is one requested file's state.
type file struct {
	fileRequest
	visible, changed map[int]bool
	dir              string // slash-separated, "." at the root
	skipped          string
	decls            []decl
	snippets         []snippet
}

// unit is a loaded file and its package.
type unit struct {
	pkg  *packages.Package
	file *ast.File
}

func run(ctx context.Context, req request) response {
	j := &job{root: req.Root, real: req.Root, src: newSources(), rels: map[string]relPath{}}
	if real, err := filepath.EvalSymlinks(req.Root); err == nil {
		j.real = real
	}
	if data, err := os.ReadFile(filepath.Join(req.Root, "go.mod")); err == nil {
		j.module = modfile.ModulePath(data)
	}
	files := j.prepare(req.Files)
	h, err := j.scan(ctx, files)
	if err != nil {
		j.errorf("scan: %v", err)
	}
	dirs := loadDirs(files)
	pkgs, err := j.load(ctx, append(dirs, h.candidates(dirs)...))
	if err != nil {
		j.errorf("load: %v", err)
		for _, f := range files {
			if f.skipped == "" {
				f.skipped = "load: " + j.clean(err.Error())
			}
		}
		return j.response(files)
	}
	idx := j.index(pkgs)
	requested := map[string]*file{}
	for _, f := range files {
		requested[f.Path] = f
	}
	for _, f := range files {
		if f.skipped != "" {
			continue
		}
		u, ok := idx[f.Path]
		if !ok || u.pkg.TypesInfo == nil {
			f.skipped = "load: no package for the file"
			continue
		}
		f.snippets = append(j.definitions(f, u), j.callers(f, idx, h, requested)...)
	}
	return j.response(files)
}

// prepare sorts the requested files and parses the ones to load, finding their
// changed funcs. A changed _test.go is skipped alone: its directory still loads
// for its siblings.
func (j *job) prepare(reqs []fileRequest) []*file {
	reqs = slices.Clone(reqs)
	slices.SortFunc(reqs, func(a, b fileRequest) int { return strings.Compare(a.Path, b.Path) })
	var dirs []string
	files := make([]*file, 0, len(reqs))
	for _, r := range reqs {
		f := &file{fileRequest: r, visible: set(r.Visible), changed: set(r.Changed), dir: path.Dir(r.Path)}
		files = append(files, f)
		switch {
		case strings.HasSuffix(r.Path, "_test.go"):
			f.skipped = "test file"
		case !j.builds(r.Path):
			f.skipped = "build-constrained"
		case !importable(f.dir):
			f.skipped = "load: not an importable directory"
		case !slices.Contains(dirs, f.dir) && len(dirs) == maxDirs:
			f.skipped = "package cap"
		default:
			if !slices.Contains(dirs, f.dir) {
				dirs = append(dirs, f.dir)
			}
			src, err := j.src.get(j.abs(r.Path))
			if err != nil {
				f.skipped = "load: " + j.clean(err.Error())
				continue
			}
			f.decls = j.changedDecls(f, src)
		}
	}
	return files
}

// importable reports whether "./dir" names that one directory as a go list
// pattern: "..." anywhere in it is a wildcard, so a PR directory named "..."
// would load, and type-check from source, the whole module.
func importable(dir string) bool {
	return !strings.Contains(dir, "...")
}

// loadDirs are the directories of the files to load, in path order.
func loadDirs(files []*file) []string {
	var dirs []string
	for _, f := range files {
		if f.skipped == "" && !slices.Contains(dirs, f.dir) {
			dirs = append(dirs, f.dir)
		}
	}
	return dirs
}

// load type-checks the changed and candidate directories in one packages.Load.
func (j *job) load(ctx context.Context, dirs []string) ([]*packages.Package, error) {
	if len(dirs) == 0 {
		return nil, nil
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	limit := maxParseFiles
	capErr := fmt.Errorf("parse file cap: more than %d files", limit)
	var parsed atomic.Int64
	cfg := &packages.Config{
		Dir: j.root,
		// Cancel/deadline propagation: Context kills the underlying `go list`
		// invocation, so a slow or hung load aborts on timeout instead of pinning
		// the worker indefinitely.
		Context: ctx,
		// Deliberately NO NeedDeps: it type-checks the entire transitive import
		// graph (third-party + stdlib) in-process, which OOM-kills the pod on a
		// real monorepo. Without it, the target package is type-checked using
		// its imports' compiled export data — bounded memory — which still
		// resolves same-package cross-file symbols (the bulk of review-useful
		// context). Cross-package definitions degrade to export-data positions
		// and are dropped when no source file is available, exactly the existing
		// def.Filename=="" skip.
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax |
			packages.NeedTypes | packages.NeedTypesInfo | packages.NeedImports,
		Fset:  token.NewFileSet(),
		Tests: false,
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			if parsed.Add(1) > limit {
				cancel(capErr)
				// A parser error, which x/tools expects from ParseFile.
				return nil, scanner.ErrorList{{Pos: token.Position{Filename: filename}, Msg: capErr.Error()}}
			}
			return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
		},
	}
	patterns := make([]string, len(dirs))
	for i, d := range dirs {
		patterns[i] = "./" + d
		if d == "." {
			patterns[i] = "."
		}
	}
	pkgs, err := packages.Load(cfg, patterns...)
	if errors.Is(context.Cause(ctx), capErr) {
		return nil, capErr
	}
	return pkgs, err
}

// index maps each loaded file's repo-relative path to it, and keeps the
// packages' errors.
func (j *job) index(pkgs []*packages.Package) map[string]unit {
	idx := map[string]unit{}
	for _, p := range pkgs {
		for _, e := range p.Errors {
			j.errs = append(j.errs, j.clean(e.Error()))
		}
		for _, f := range p.Syntax {
			if rel, ok := j.rel(p.Fset.File(f.FileStart).Name()); ok {
				idx[rel] = unit{p, f}
			}
		}
	}
	return idx
}

func (j *job) response(files []*file) response {
	errs := slices.Compact(slices.Sorted(slices.Values(j.errs)))
	resp := response{
		Go:     runtime.Version(),
		Errors: append([]string{}, errs[:min(len(errs), maxErrors)]...),
		Files:  make([]fileResult, 0, len(files)),
	}
	for _, f := range files {
		resp.Files = append(resp.Files, fileResult{Path: f.Path, Skipped: f.skipped, Snippets: dedupe(f.snippets)})
	}
	return resp
}

// dedupe drops a snippet whose (path, start) an earlier one has.
func dedupe(snippets []snippet) []snippet {
	type key struct {
		path  string
		start int
	}
	seen := map[key]bool{}
	out := []snippet{}
	for _, s := range snippets {
		k := key{s.Path, s.StartLine}
		if !seen[k] {
			seen[k] = true
			out = append(out, s)
		}
	}
	return out
}

func (j *job) abs(rel string) string {
	return filepath.Join(j.root, filepath.FromSlash(rel))
}

// rel is name's slash-separated path under the root after resolving symlinks, so
// GOROOT and the module cache never count; false when it isn't under it. A name
// starting with the module path is a main-module file in -trimpath export data.
func (j *job) rel(name string) (string, bool) {
	if r, ok := j.rels[name]; ok {
		return r.path, r.ok
	}
	var r relPath
	file := name
	if rest, ok := strings.CutPrefix(name, j.module+"/"); ok && j.module != "" {
		file = j.abs(rest)
	}
	if real, err := filepath.EvalSymlinks(file); err == nil {
		if p, err := filepath.Rel(j.real, real); err == nil && filepath.IsLocal(p) {
			r = relPath{filepath.ToSlash(p), true}
		}
	}
	j.rels[name] = r
	return r.path, r.ok
}

// builds reports whether rel is in the build for this platform (GOOS, GOARCH,
// CGO_ENABLED, build tags), as `go list` will see it. A file that can't be read
// counts as building, so its read error is what's reported.
func (j *job) builds(rel string) bool {
	dir, name := path.Split(rel)
	ok, err := build.Default.MatchFile(j.abs(dir), name)
	return ok || err != nil
}

func (j *job) errorf(format string, args ...any) {
	j.errs = append(j.errs, j.clean(fmt.Sprintf(format, args...)))
}

// clean shows the root as "." in a message, the resolved root first, since the
// requested one may be a suffix of it (macOS's /var is /private/var).
func (j *job) clean(msg string) string {
	roots := []string{j.root, j.real}
	slices.SortFunc(roots, func(a, b string) int { return cmp.Compare(len(b), len(a)) })
	for _, r := range roots {
		msg = strings.ReplaceAll(msg, r+string(filepath.Separator), "./")
		msg = strings.ReplaceAll(msg, r, ".")
	}
	return msg
}

func set(lines []int) map[int]bool {
	s := make(map[int]bool, len(lines))
	for _, n := range lines {
		s[n] = true
	}
	return s
}
