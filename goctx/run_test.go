package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func sampleRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs("testdata/sample")
	if err != nil {
		t.Fatal(err)
	}
	return root
}

// lines is start..end.
func lines(start, end int) []int {
	var out []int
	for n := start; n <= end; n++ {
		out = append(out, n)
	}
	return out
}

// sampleWith is a temporary copy of the sample module plus extra files. Hostile
// fixtures (a wildcard directory name, a type error) live here rather than in
// testdata, where they would break `go build ./...` of the sample module.
func sampleWith(t *testing.T, extra map[string]string) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "sample")
	if err := os.CopyFS(root, os.DirFS(sampleRoot(t))); err != nil {
		t.Fatal(err)
	}
	for rel, src := range extra {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// wildcardSample adds a package in "x...", which go list would read as a wildcard.
func wildcardSample(t *testing.T) string {
	t.Helper()
	return sampleWith(t, map[string]string{"x.../call.go": "package x\n\nimport \"sample/clock\"\n\n" +
		"// Call is in a directory whose name is a go list wildcard.\nfunc Call() int {\n\treturn clock.Add(1, 2)\n}\n"})
}

func runSample(t *testing.T, files ...fileRequest) response {
	t.Helper()
	return runAt(t, sampleRoot(t), files...)
}

func runAt(t *testing.T, root string, files ...fileRequest) response {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	return run(ctx, request{Root: root, DeadlineMS: 60000, Files: files})
}

func result(t *testing.T, resp response, path string) fileResult {
	t.Helper()
	for _, f := range resp.Files {
		if f.Path == path {
			return f
		}
	}
	t.Fatalf("no result for %s in %+v", path, resp)
	return fileResult{}
}

// where is a snippet without its text, for exact comparison.
func where(snippets []snippet, kind string) []string {
	var out []string
	for _, s := range snippets {
		if kind == "" || s.Kind == kind {
			out = append(out, fmt.Sprintf("%s %s %s:%d-%d", s.Kind, s.Symbol, s.Path, s.StartLine, s.EndLine))
		}
	}
	return out
}

func joined(snippets []snippet) string {
	var b strings.Builder
	for _, s := range snippets {
		b.WriteString(s.Text + "\n")
	}
	return b.String()
}

// handler.go as an added file (its patch covers lines 1-11). Notifier lives in
// another package, so it comes from export data: this fails when x/tools can't
// read what the running Go's `go list` writes. The client runs goctx with
// -trimpath (so the build cache survives a new checkout directory), where
// export data names notif.go by its import path, "sample/notif/notif.go".
func TestDefinitions_CrossPackageInterface(t *testing.T) {
	for _, flags := range []string{"", "-trimpath"} {
		t.Run("GOFLAGS="+flags, func(t *testing.T) {
			t.Setenv("GOFLAGS", flags)
			resp := runSample(t, fileRequest{Path: "handler.go", Visible: lines(1, 11), Changed: lines(1, 11)})
			got := result(t, resp, "handler.go")
			if got.Skipped != "" || len(resp.Errors) != 0 {
				t.Fatalf("skipped %q, errors %v", got.Skipped, resp.Errors)
			}
			if want := []string{"definition Notifier notif/notif.go:3-8"}; !slices.Equal(where(got.Snippets, ""), want) {
				t.Fatalf("snippets %v, want %v", where(got.Snippets, ""), want)
			}
			for _, want := range []string{"type Notifier interface", "Send(", "Close(", "// Notifier sends messages."} {
				if !strings.Contains(got.Snippets[0].Text, want) {
					t.Errorf("missing %q in\n%s", want, got.Snippets[0].Text)
				}
			}
		})
	}
}

// A //line directive moves fset.Position's line (b.go:100000) but not the
// physical one goctx slices source by and the diff numbers lines in.
func TestDefinitions_LineDirective(t *testing.T) {
	resp := runSample(t, fileRequest{Path: "linedir/a.go", Visible: []int{3}, Changed: []int{3}})
	want := []string{
		"definition I linedir/b.go:3-4",
		"definition M linedir/b.go:4-4",
	}
	if got := where(result(t, resp, "linedir/a.go").Snippets, ""); !slices.Equal(got, want) {
		t.Errorf("snippets %v, want %v", got, want)
	}
}

// A patch touching only Send; same-file siblings are not pulled in. Its receiver
// type, declared off the visible lines, is a definition; the interface method it
// calls is one line; the changed method itself is never echoed back.
func TestDefinitions_ChangedMethodNotEchoed(t *testing.T) {
	resp := runSample(t, fileRequest{Path: "handler.go", Visible: []int{14, 15, 16}, Changed: []int{15}})
	got := result(t, resp, "handler.go")
	want := []string{
		"definition Handler handler.go:5-8",
		"definition Send notif/notif.go:6-6",
	}
	if !slices.Equal(where(got.Snippets, ""), want) {
		t.Fatalf("snippets %v, want %v", where(got.Snippets, ""), want)
	}
	if got.Snippets[1].Text != "\tSend(msg string) error" {
		t.Errorf("interface method text %q", got.Snippets[1].Text)
	}
	if text := joined(got.Snippets); strings.Contains(text, "func (h *Handler) Send") {
		t.Errorf("the changed method was echoed back:\n%s", text)
	}
}

// Types, then funcs; within a kind a use on an added line (Handler, line 14)
// before an earlier use on a context line (Notifier, line 10).
func TestDefinitions_Order(t *testing.T) {
	resp := runSample(t, fileRequest{Path: "handler.go", Visible: []int{10, 11, 12, 14, 15, 16}, Changed: []int{14}})
	want := []string{
		"definition Handler handler.go:5-8",
		"definition Notifier notif/notif.go:3-8",
		"definition Send notif/notif.go:6-6",
	}
	if got := where(result(t, resp, "handler.go").Snippets, ""); !slices.Equal(got, want) {
		t.Errorf("snippets %v, want %v", got, want)
	}
}

// A same-file helper referenced by a changed line but declared outside the diff.
func TestDefinitions_SameFileReferencedDef(t *testing.T) {
	resp := runSample(t, fileRequest{Path: "util.go", Visible: []int{10, 11, 12}, Changed: []int{11}})
	got := result(t, resp, "util.go")
	if want := []string{"definition formatMsg util.go:3-8"}; !slices.Equal(where(got.Snippets, ""), want) {
		t.Fatalf("snippets %v, want %v", where(got.Snippets, ""), want)
	}
	if text := joined(got.Snippets); strings.Contains(text, "func Greet") {
		t.Errorf("the changed function was echoed back:\n%s", text)
	}
}

func TestCallers(t *testing.T) {
	resp := runSample(t,
		// InRange's and Add's bodies.
		fileRequest{Path: "clock/clock.go", Visible: []int{14, 15, 16, 19, 20, 21}, Changed: []int{15, 20}},
		// getUser's body: unexported, so it lifts to GetUser.
		fileRequest{Path: "store/store.go", Visible: []int{18, 19, 20}, Changed: []int{19}},
	)
	tests := []struct {
		name, path string
		want       []string
	}{
		{"direct call, then a call through an interface *ranger implements", "clock/clock.go", []string{
			"caller Add app/direct.go:6-9",
			"caller InRange app/iface.go:6-8",
		}},
		{"the exported wrapper's caller through an interface, then the unexported in-package caller",
			"store/store.go", []string{
				"caller GetUser app/refund.go:6-12",
				"caller getUser store/store.go:12-16",
			}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := result(t, resp, tt.path)
			if !slices.Equal(where(got.Snippets, "caller"), tt.want) {
				t.Errorf("callers %v, want %v", where(got.Snippets, "caller"), tt.want)
			}
			if strings.Contains(joined(got.Snippets), "Generated") {
				t.Error("a generated file is a caller")
			}
		})
	}
	if got := result(t, resp, "clock/clock.go").Snippets[1].Text; got != "func Total(a, b int) int {\n\tsum := clock.Add(a, b)\n\treturn sum\n}" {
		t.Errorf("site text %q", got)
	}
}

// A use inside an earlier site's window (use.go:8 in 4-20) takes no slot, so
// the next use past it (:26) is shown instead of the same lines twice.
func TestCallers_SiteInsideAnEarlierOneSkipped(t *testing.T) {
	resp := runSample(t, fileRequest{Path: "tick/tick.go", Visible: []int{4, 5, 6}, Changed: []int{5}})
	want := []string{
		"caller Tick tick/use.go:4-20",
		"caller Tick tick/use.go:24-27",
	}
	if got := where(result(t, resp, "tick/tick.go").Snippets, "caller"); !slices.Equal(got, want) {
		t.Errorf("callers %v, want %v", got, want)
	}
}

func TestCallers_VisibleSitesSkipped(t *testing.T) {
	resp := runSample(t,
		fileRequest{Path: "app/direct.go", Visible: lines(6, 9), Changed: []int{8}},
		fileRequest{Path: "clock/clock.go", Visible: []int{19, 20, 21}, Changed: []int{20}},
	)
	if got := where(result(t, resp, "clock/clock.go").Snippets, "caller"); len(got) != 0 {
		t.Errorf("callers %v: the call on a visible line of app/direct.go is no caller", got)
	}
}

// A func line is never a hit (minute.go declares another Add), nor is anything
// in a generated file (gen.go calls clock.Add) or in x.../, which as a pattern
// would load every directory starting with x: app's one hit is direct.go.
func TestScan_FuncLinesAndGeneratedFilesAreNoHits(t *testing.T) {
	root := wildcardSample(t)
	j := &job{root: root, real: root, src: newSources(), rels: map[string]relPath{}}
	files := []*file{{decls: []decl{{path: "clock/clock.go", dir: "clock", name: "Add", line: 19}}}}
	h, err := j.scan(context.Background(), files)
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string]int{"app": 1}; !maps.Equal(h.count, want) {
		t.Errorf("hits %v, want %v", h.count, want)
	}
	if want := map[string]bool{"app/direct.go": true}; !maps.Equal(h.files, want) {
		t.Errorf("files %v, want %v", h.files, want)
	}
}

func TestSkips(t *testing.T) {
	files := []fileRequest{
		{Path: "clock/clock_test.go", Visible: []int{1}, Changed: []int{1}},
		{Path: "clock/clock_windows.go", Visible: []int{4}, Changed: []int{4}},
		{Path: "clock/clock.go", Visible: []int{19, 20, 21}, Changed: []int{20}},
		{Path: "x.../call.go", Visible: []int{7}, Changed: []int{7}},
	}
	// Eight more directories that don't exist: seven fill the cap, one is past it.
	for _, d := range "abcdefgh" {
		files = append(files, fileRequest{Path: string(d) + "/x.go", Visible: []int{1}, Changed: []int{1}})
	}
	resp := runAt(t, wildcardSample(t), files...)
	want := map[string]string{
		"a/x.go":                 "load: open ./a/x.go: no such file or directory",
		"clock/clock.go":         "",
		"clock/clock_test.go":    "test file",
		"clock/clock_windows.go": "build-constrained",
		"g/x.go":                 "load: open ./g/x.go: no such file or directory",
		"h/x.go":                 "package cap",
		"x.../call.go":           "load: not an importable directory",
	}
	var paths []string
	for _, f := range resp.Files {
		paths = append(paths, f.Path)
		if w, ok := want[f.Path]; ok && f.Skipped != w {
			t.Errorf("%s: skipped %q, want %q", f.Path, f.Skipped, w)
		}
	}
	if !slices.IsSorted(paths) || len(paths) != len(files) {
		t.Errorf("paths %v: want all, sorted", paths)
	}
	if got := where(result(t, resp, "clock/clock.go").Snippets, "caller"); !slices.Equal(got, []string{"caller Add app/direct.go:6-9"}) {
		t.Errorf("a changed test file's sibling lost its callers: %v", got)
	}
}

func TestParseFileCap(t *testing.T) {
	defer func(n int64) { maxParseFiles = n }(maxParseFiles)
	maxParseFiles = 1
	resp := runSample(t, fileRequest{Path: "handler.go", Visible: lines(1, 11), Changed: lines(1, 11)})
	const want = "load: parse file cap: more than 1 files"
	if got := result(t, resp, "handler.go"); got.Skipped != want || len(got.Snippets) != 0 {
		t.Errorf("skipped %q with %d snippets, want %q", got.Skipped, len(got.Snippets), want)
	}
	if !slices.Contains(resp.Errors, want) {
		t.Errorf("errors %v, want %q", resp.Errors, want)
	}
}

func TestNoRootInMessages(t *testing.T) {
	root := sampleWith(t, map[string]string{"broken/broken.go": "package broken\n\n" +
		"// Count is wrong on purpose: a test wants a type error.\nvar Count int = \"three\"\n"})
	resp := runAt(t, root,
		fileRequest{Path: "broken/broken.go", Visible: []int{4}, Changed: []int{4}},
		fileRequest{Path: "missing/x.go", Visible: []int{1}, Changed: []int{1}},
	)
	typeErr := func(e string) bool { return strings.HasPrefix(e, "./broken/broken.go:4:17: cannot use") }
	if !slices.ContainsFunc(resp.Errors, typeErr) {
		t.Errorf("errors %v: want the type error at ./broken/broken.go:4", resp.Errors)
	}
	out, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	real, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []string{root, real} {
		if bytes.Contains(out, []byte(r)) {
			t.Errorf("response shows the root %s:\n%s", r, out)
		}
	}
}
