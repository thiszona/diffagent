package main

import (
	"bytes"
	"cmp"
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"io/fs"
	"maps"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// Caller bounds; the per-name, per-file and per-changed-file caps and the site
// window are the TS context's (src/context/ts.rs).
const (
	maxNames        = 8  // changed funcs per file
	maxLifts        = 2  // exported wrappers per unexported changed func
	maxCandidates   = 6  // caller directories loaded beyond the changed ones
	maxFilesPerName = 6  // calling files per changed func
	maxSitesPerFile = 2  // sites per calling file
	maxCallers      = 8  // caller snippets per changed file
	siteBefore      = 2  // lines kept above a site
	siteAfter       = 15 // and below it
	maxScanBytes    = 512 << 10
)

// decl is a changed top-level func or method, or an exported wrapper lifted from
// an unexported one, found by syntax before the load.
type decl struct {
	path string // repo-relative file
	dir  string
	name string
	line int // of the name
}

// changedDecls are the top-level funcs and methods overlapping the changed
// lines (≤maxNames), then the wrappers each unexported one lifts to.
func (j *job) changedDecls(f *file, src *source) []decl {
	var out []decl
	for _, d := range src.file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || !overlaps(f.changed, j.src.line(fd.Pos()), j.src.line(fd.End())) {
			continue
		}
		if len(out) == maxNames {
			break
		}
		out = append(out, decl{f.Path, f.dir, fd.Name.Name, j.src.line(fd.Name.Pos())})
	}
	for _, d := range out {
		if !token.IsExported(d.name) {
			for _, w := range j.lift(d, src.file.Name.Name) {
				if !slices.Contains(out, w) {
					out = append(out, w)
				}
			}
		}
	}
	return out
}

func overlaps(lines map[int]bool, start, end int) bool {
	for n := range lines {
		if n >= start && n <= end {
			return true
		}
	}
	return false
}

// lift finds, one hop, the exported top-level funcs and methods of d's package
// whose bodies name it (≤maxLifts, by file then line). A package that wraps an
// unexported twin in a decorator has its callers call the wrapper: a change to
// internalGetUserByID is called as InternalGetUserByID.
func (j *job) lift(d decl, pkg string) []decl {
	entries, err := os.ReadDir(j.abs(d.dir))
	if err != nil {
		return nil
	}
	var out []decl
	for _, e := range entries {
		name := e.Name()
		rel := path.Join(d.dir, name)
		if !e.Type().IsRegular() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") || !j.builds(rel) {
			continue
		}
		src, err := j.src.get(j.abs(rel))
		if err != nil || src.file.Name.Name != pkg {
			continue
		}
		for _, top := range src.file.Decls {
			fd, ok := top.(*ast.FuncDecl)
			if !ok || !fd.Name.IsExported() || fd.Body == nil || !mentions(fd.Body, d.name) {
				continue
			}
			out = append(out, decl{rel, d.dir, fd.Name.Name, j.src.line(fd.Name.Pos())})
			if len(out) == maxLifts {
				return out
			}
		}
	}
	return out
}

func mentions(n ast.Node, name string) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return !found
	})
	return found
}

// hits is the text scan's result, which picks the caller directories to load.
type hits struct {
	count  map[string]int  // hits per directory
	files  map[string]bool // files with a hit
	ranked []string        // directories by hits, most first, then path
	rank   map[string]int  // each directory's position in ranked
}

// scan finds candidate callers by text before anything loads, in hand-written
// .go files outside tests, vendor, testdata, dot and underscore directories,
// directories no pattern can name (importable) and nested modules: ".Name(" anywhere for an exported name, "Name(" in the changed
// func's own directory. A func line is never a hit, so a directory of
// same-named declarations (every Handle in app/command) can't take the slots.
func (j *job) scan(ctx context.Context, files []*file) (hits, error) {
	h := hits{count: map[string]int{}, files: map[string]bool{}, rank: map[string]int{}}
	dirs := map[string]map[string]bool{} // name -> directories where a bare call counts
	for _, f := range files {
		for _, d := range f.decls {
			if dirs[d.name] == nil {
				dirs[d.name] = map[string]bool{}
			}
			dirs[d.name][d.dir] = true
		}
	}
	if len(dirs) == 0 {
		return h, nil
	}
	names := slices.Sorted(maps.Keys(dirs))
	for i, n := range names {
		names[i] = regexp.QuoteMeta(n)
	}
	re := regexp.MustCompile(`\b(` + strings.Join(names, "|") + `)\(`)
	err := filepath.WalkDir(j.root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // an unreadable entry has no hits
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		rel, _ := filepath.Rel(j.root, p)
		rel = filepath.ToSlash(rel)
		name := d.Name()
		if d.IsDir() {
			if rel == "." {
				return nil
			}
			if name == "vendor" || name == "testdata" || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") ||
				!importable(rel) {
				return filepath.SkipDir
			}
			if _, err := os.Lstat(filepath.Join(p, "go.mod")); err == nil {
				return filepath.SkipDir // a nested module
			}
			return nil
		}
		if !d.Type().IsRegular() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") ||
			strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
			return nil
		}
		if info, err := d.Info(); err != nil || info.Size() > maxScanBytes {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil || !re.Match(data) || generated(data) {
			return nil
		}
		dir := path.Dir(rel)
		if n := countHits(data, re, dir, dirs); n > 0 {
			h.count[dir] += n
			h.files[rel] = true
		}
		return nil
	})
	h.ranked = slices.Collect(maps.Keys(h.count))
	slices.SortFunc(h.ranked, func(a, b string) int {
		return cmp.Or(cmp.Compare(h.count[b], h.count[a]), cmp.Compare(a, b))
	})
	for i, d := range h.ranked {
		h.rank[d] = i
	}
	return h, err
}

func countHits(src []byte, re *regexp.Regexp, dir string, dirs map[string]map[string]bool) int {
	n := 0
	for line := range bytes.Lines(src) {
		if bytes.HasPrefix(bytes.TrimSpace(line), []byte("func ")) {
			continue
		}
		for _, m := range re.FindAllSubmatchIndex(line, -1) {
			name := string(line[m[2]:m[3]])
			dotted := m[2] > 0 && line[m[2]-1] == '.'
			if dotted && token.IsExported(name) || dirs[name][dir] {
				n++
			}
		}
	}
	return n
}

var generatedRE = regexp.MustCompile(`^// Code generated .* DO NOT EDIT\.$`)

// generated reports the marker before the package clause
// (https://go.dev/s/generatedcode).
func generated(src []byte) bool {
	for line := range bytes.Lines(src) {
		line = bytes.TrimRight(line, "\r\n")
		if bytes.HasPrefix(line, []byte("package ")) {
			return false
		}
		if generatedRE.Match(line) {
			return true
		}
	}
	return false
}

// candidates are the best-ranked directories beyond the loaded ones.
func (h hits) candidates(loaded []string) []string {
	var out []string
	for _, d := range h.ranked {
		if len(out) == maxCandidates {
			break
		}
		if !slices.Contains(loaded, d) {
			out = append(out, d)
		}
	}
	return out
}

// target is a changed func, or a lifted wrapper, as the load typed it.
type target struct {
	fn *types.Func
	// named is a method's receiver type, for calls through an interface it
	// implements; nil for funcs and generic receivers.
	named *types.Named
}

// calledBy reports whether a use of fn reaches t: t itself, or a same-name
// interface method that t's receiver implements (callers often hold an
// interface, not the concrete type).
func (t target) calledBy(fn *types.Func) bool {
	if fn == t.fn || fn.Name() == t.fn.Name() && fn.FullName() == t.fn.FullName() {
		return true
	}
	if t.named == nil || fn.Name() != t.fn.Name() {
		return false
	}
	recv := fn.Signature().Recv()
	if recv == nil {
		return false
	}
	iface, ok := recv.Type().Underlying().(*types.Interface)
	return ok && (types.Implements(t.named, iface) || types.Implements(types.NewPointer(t.named), iface))
}

func (j *job) targets(f *file, idx map[string]unit) []target {
	var out []target
	for _, d := range f.decls {
		u, ok := idx[d.path]
		if !ok || u.pkg.TypesInfo == nil {
			continue
		}
		for _, top := range u.file.Decls {
			fd, ok := top.(*ast.FuncDecl)
			if !ok || fd.Name.Name != d.name || u.pkg.Fset.PositionFor(fd.Name.Pos(), false).Line != d.line {
				continue
			}
			if fn, ok := u.pkg.TypesInfo.Defs[fd.Name].(*types.Func); ok {
				out = append(out, target{fn, receiver(fn)})
			}
		}
	}
	return out
}

func receiver(fn *types.Func) *types.Named {
	recv := fn.Signature().Recv()
	if recv == nil {
		return nil
	}
	t := types.Unalias(recv.Type())
	if p, ok := t.(*types.Pointer); ok {
		t = types.Unalias(p.Elem())
	}
	// Implements is unspecified for an uninstantiated generic type.
	if n, ok := t.(*types.Named); ok && n.TypeParams().Len() == 0 {
		return n
	}
	return nil
}

// site is a confirmed use of a target in a calling file.
type site struct {
	name               string
	line               int
	declStart, declEnd int // the enclosing top-level declaration
}

func sites(u unit, targets []target) []site {
	fset, info := u.pkg.Fset, u.pkg.TypesInfo
	var out []site
	for _, d := range u.file.Decls {
		start, end := fset.PositionFor(d.Pos(), false).Line, fset.PositionFor(d.End(), false).Line
		ast.Inspect(d, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			fn, ok := info.Uses[id].(*types.Func)
			if !ok {
				return true
			}
			line := fset.PositionFor(id.Pos(), false).Line
			for _, t := range targets {
				if t.calledBy(fn.Origin()) {
					if len(out) == 0 || out[len(out)-1].line != line {
						out = append(out, site{t.fn.Name(), line, start, end})
					}
					break
				}
			}
			return true
		})
	}
	return out
}

// callers are the confirmed uses of what f changed, in the scan's directory rank,
// then path, then line, skipping sites on a requested file's visible lines and
// inside an earlier site of the same file; each is the use line -2 to +15,
// clipped to its top-level declaration.
func (j *job) callers(f *file, idx map[string]unit, h hits, requested map[string]*file) []snippet {
	targets := j.targets(f, idx)
	if len(targets) == 0 {
		return nil
	}
	var files []string
	for rel := range h.files {
		if _, ok := idx[rel]; ok {
			files = append(files, rel)
		}
	}
	slices.SortFunc(files, func(a, b string) int {
		return cmp.Or(cmp.Compare(h.rank[path.Dir(a)], h.rank[path.Dir(b)]), cmp.Compare(a, b))
	})
	perName := map[string]map[string]bool{}
	var out []snippet
	for _, rel := range files {
		n := 0
		for _, s := range sites(idx[rel], targets) {
			if req := requested[rel]; req != nil && req.visible[s.line] {
				continue
			}
			if slices.ContainsFunc(out, func(c snippet) bool {
				return c.Path == rel && s.line >= c.StartLine && s.line <= c.EndLine
			}) {
				continue // inside a site already shown
			}
			if n == maxSitesPerFile || len(out) == maxCallers {
				break
			}
			seen := perName[s.name]
			if seen == nil {
				seen = map[string]bool{}
				perName[s.name] = seen
			}
			if !seen[rel] && len(seen) == maxFilesPerName {
				continue
			}
			src, err := j.src.get(j.abs(rel))
			if err != nil {
				break
			}
			seen[rel] = true
			start, end := max(s.line-siteBefore, s.declStart), min(s.line+siteAfter, s.declEnd)
			out = append(out, snippet{
				Kind:      "caller",
				Symbol:    s.name,
				Path:      rel,
				StartLine: start,
				EndLine:   end,
				Text:      src.text(start, end),
			})
			n++
		}
		if len(out) == maxCallers {
			break
		}
	}
	return out
}
