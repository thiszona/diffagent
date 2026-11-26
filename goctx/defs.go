package main

import (
	"cmp"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"slices"
	"strings"
)

// maxSnippetLines bounds a definition's text, doc comment included.
const maxSnippetLines = 40

// Definition kinds, in snippet order.
const (
	kindType = iota
	kindFunc
	kindValue
)

// sources parses files with go/parser for snippet text, once each.
type sources struct {
	fset  *token.FileSet
	files map[string]*source
}

type source struct {
	file  *ast.File
	lines []string
	err   error
}

func newSources() *sources {
	return &sources{fset: token.NewFileSet(), files: map[string]*source{}}
}

func (s *sources) get(name string) (*source, error) {
	if src, ok := s.files[name]; ok {
		return src, src.err
	}
	src := &source{}
	data, err := os.ReadFile(name)
	if err == nil {
		src.lines = strings.Split(string(data), "\n")
		src.file, err = parser.ParseFile(s.fset, name, data, parser.ParseComments|parser.SkipObjectResolution)
	}
	src.err = err
	s.files[name] = src
	return src, err
}

// line is p's physical line: a //line directive must not move it off the diff's
// lines or past the end of the file.
func (s *sources) line(p token.Pos) int {
	return s.fset.PositionFor(p, false).Line
}

// text is lines start..end clipped to the file, "" when none is in it: export
// data's positions follow //line directives, so they can lie past the end.
func (src *source) text(start, end int) string {
	start, end = max(start, 1), min(end, len(src.lines))
	if start > end {
		return ""
	}
	return strings.Join(src.lines[start-1:end], "\n")
}

// use is where a definition is first used on the visible lines.
type use struct {
	obj       types.Object
	added     bool
	line, col int
}

// definitions are the package-level objects and methods the file's visible lines
// use, declared under the root and off those lines: types, then funcs and
// methods, then constants and variables; within a kind by first use, a use on an
// added line before one on a context line, then by (path, line).
func (j *job) definitions(f *file, u unit) []snippet {
	info, fset := u.pkg.TypesInfo, u.pkg.Fset
	first := map[types.Object]*use{}
	var order []*use
	ast.Inspect(u.file, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		at := fset.PositionFor(id.Pos(), false)
		obj := origin(info.Uses[id])
		if !f.visible[at.Line] || kindOf(obj) < 0 {
			return true
		}
		added := f.changed[at.Line]
		if prev, ok := first[obj]; ok {
			if added && !prev.added {
				*prev = use{obj, true, at.Line, at.Column}
			}
			return true
		}
		first[obj] = &use{obj, added, at.Line, at.Column}
		order = append(order, first[obj])
		return true
	})
	type def struct {
		use
		kind int
		s    snippet
	}
	var defs []def
	for _, d := range order {
		at := fset.PositionFor(d.obj.Pos(), false)
		rel, ok := j.rel(at.Filename)
		if !ok || rel == f.Path && f.visible[at.Line] {
			continue
		}
		if s, ok := j.definition(d.obj, at, rel); ok {
			defs = append(defs, def{*d, kindOf(d.obj), s})
		}
	}
	slices.SortStableFunc(defs, func(a, b def) int {
		return cmp.Or(
			cmp.Compare(a.kind, b.kind),
			cmpTrueFirst(a.added, b.added),
			cmp.Compare(a.line, b.line),
			cmp.Compare(a.col, b.col),
			cmp.Compare(a.s.Path, b.s.Path),
			cmp.Compare(a.s.StartLine, b.s.StartLine),
		)
	})
	out := make([]snippet, len(defs))
	for i, d := range defs {
		out[i] = d.s
	}
	return out
}

// definition renders obj's declaration from rel, the file at names: its source
// and doc comment, ≤40 lines; an interface method is its one line.
func (j *job) definition(obj types.Object, at token.Position, rel string) (snippet, bool) {
	src, err := j.src.get(j.abs(rel))
	if err != nil {
		return snippet{}, false
	}
	start, end := at.Line, at.Line
	if !isInterfaceMethod(obj) {
		var ok bool
		if start, end, ok = j.src.declLines(src, at.Line); !ok {
			return snippet{}, false
		}
	}
	end = min(end, start+maxSnippetLines-1, len(src.lines))
	if start > end {
		return snippet{}, false // a line from export data, past the file's end
	}
	return snippet{
		Kind:      "definition",
		Symbol:    obj.Name(),
		Path:      rel,
		StartLine: start,
		EndLine:   end,
		Text:      src.text(start, end),
	}, true
}

// declLines spans the top-level declaration holding line, doc comment included:
// a func, an ungrouped type, var or const, or one spec of a group.
func (s *sources) declLines(src *source, line int) (start, end int, ok bool) {
	for _, d := range src.file.Decls {
		if line < s.line(d.Pos()) || line > s.line(d.End()) {
			continue
		}
		switch d := d.(type) {
		case *ast.FuncDecl:
			return s.withDoc(d.Doc, d.Pos()), s.line(d.End()), true
		case *ast.GenDecl:
			if !d.Lparen.IsValid() {
				return s.withDoc(d.Doc, d.Pos()), s.line(d.End()), true
			}
			for _, spec := range d.Specs {
				if line >= s.line(spec.Pos()) && line <= s.line(spec.End()) {
					return s.withDoc(specDoc(spec), spec.Pos()), s.line(spec.End()), true
				}
			}
		}
	}
	return 0, 0, false
}

func (s *sources) withDoc(doc *ast.CommentGroup, pos token.Pos) int {
	if doc != nil {
		return s.line(doc.Pos())
	}
	return s.line(pos)
}

func specDoc(spec ast.Spec) *ast.CommentGroup {
	switch sp := spec.(type) {
	case *ast.TypeSpec:
		return sp.Doc
	case *ast.ValueSpec:
		return sp.Doc
	}
	return nil
}

// kindOf is obj's definition kind, or -1 for what is never a definition: the
// universe, locals, parameters, fields and package names.
func kindOf(obj types.Object) int {
	if obj == nil || obj.Pkg() == nil || !obj.Pos().IsValid() {
		return -1
	}
	pkgLevel := obj.Parent() == obj.Pkg().Scope()
	switch o := obj.(type) {
	case *types.TypeName:
		if pkgLevel {
			return kindType
		}
	case *types.Func:
		if pkgLevel || o.Signature().Recv() != nil {
			return kindFunc
		}
	case *types.Const:
		if pkgLevel {
			return kindValue
		}
	case *types.Var:
		if pkgLevel && !o.IsField() {
			return kindValue
		}
	}
	return -1
}

// origin is the declared object behind an instantiation of a generic one.
func origin(obj types.Object) types.Object {
	switch o := obj.(type) {
	case *types.Func:
		return o.Origin()
	case *types.Var:
		return o.Origin()
	}
	return obj
}

func isInterfaceMethod(obj types.Object) bool {
	fn, ok := obj.(*types.Func)
	if !ok {
		return false
	}
	recv := fn.Signature().Recv()
	return recv != nil && types.IsInterface(recv.Type())
}

func cmpTrueFirst(a, b bool) int {
	switch {
	case a == b:
		return 0
	case a:
		return -1
	}
	return 1
}
