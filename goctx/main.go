// Command diffagent-goctx finds the Go context a reviewer needs and cannot see in a
// pull request's diff: the out-of-diff definitions the changed lines use, and the
// code that calls what changed. diffagent runs it once per Review:
// a JSON request on stdin, a JSON response on stdout. It type-checks with one
// packages.Load, so the memory spike lands in this process, not the worker.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"runtime/metrics"
	"strings"
	"time"
)

// maxFiles bounds one request; the client sends at most this many.
const maxFiles = 32

type request struct {
	// Root is the module root: absolute, with a go.mod.
	Root       string        `json:"root"`
	DeadlineMS int64         `json:"deadline_ms"`
	Files      []fileRequest `json:"files"`
}

type fileRequest struct {
	// Path is repo-relative.
	Path string `json:"path"`
	// Visible are the new-file lines the model is shown: added and context.
	Visible []int `json:"visible"`
	// Changed are the added lines and where removed runs sat.
	Changed []int `json:"changed"`
}

type response struct {
	// Go is this binary's toolchain, which must read the export data `go list`
	// writes.
	Go string `json:"go"`
	// Errors are load, type and scan errors, sorted, the root shown as ".".
	Errors []string     `json:"errors"`
	Files  []fileResult `json:"files"`
}

type fileResult struct {
	Path string `json:"path"`
	// Skipped says why the file got no context: "test file",
	// "build-constrained", "package cap" or "load: …"; empty when it did.
	Skipped  string    `json:"skipped"`
	Snippets []snippet `json:"snippets"`
}

// snippet is diffagent's context::Snippet.
type snippet struct {
	Kind      string `json:"kind"` // "definition" or "caller"
	Symbol    string `json:"symbol"`
	Path      string `json:"path"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Text      string `json:"text"`
}

// exitMemory is the exit code when the heap passes its hard cap, which the client
// reports as a known failure rather than a crash.
const exitMemory = 3

func main() {
	adjustOOMScore()
	if limit := debug.SetMemoryLimit(-1); limit != math.MaxInt64 {
		go watchHeap(hardCap(limit), 100*time.Millisecond, os.Stderr, os.Exit)
	}
	os.Exit(serve(os.Stdin, os.Stdout, os.Stderr))
}

// hardCap is the memory a run may reach under GOMEMLIMIT `limit`: half again as
// much. The limit is soft, so a load that keeps growing passes it and the GC only
// works harder, until the cgroup's OOM killer ends it.
func hardCap(limit int64) uint64 {
	return uint64(limit) / 2 * 3
}

// watchHeap reads the memory the runtime holds every tick, as GOMEMLIMIT counts it,
// and calls exit(exitMemory) with a line on stderr once it passes hardLimit.
func watchHeap(hardLimit uint64, tick time.Duration, stderr io.Writer, exit func(int)) {
	samples := []metrics.Sample{
		{Name: "/memory/classes/total:bytes"},
		{Name: "/memory/classes/heap/released:bytes"},
	}
	t := time.NewTicker(tick)
	defer t.Stop()
	for range t.C {
		metrics.Read(samples)
		held := samples[0].Value.Uint64() - samples[1].Value.Uint64()
		if held > hardLimit {
			fmt.Fprintf(stderr, "heap %d MiB passed its hard cap of %d MiB\n", held>>20, hardLimit>>20)
			exit(exitMemory)
			return
		}
	}
}

// adjustOOMScore makes this process, and the go and compile processes it starts
// (they inherit it), the first a cgroup OOM kills, so a big type-check never takes
// the worker down. Best effort: the file exists only on Linux.
func adjustOOMScore() {
	f, err := os.OpenFile("/proc/self/oom_score_adj", os.O_WRONLY, 0)
	if err != nil {
		return
	}
	_, _ = f.WriteString("1000")
	_ = f.Close()
}

// serve answers one request and returns the exit code: 2 for a bad request (a
// bug in the caller), 1 for a panic, else 0, with every failure past that in the
// response. The heap watchdog exits 3 on its own.
func serve(stdin io.Reader, stdout, stderr io.Writer) (code int) {
	defer recovered(stderr, &code)
	req, err := decode(stdin)
	if err != nil {
		fmt.Fprintf(stderr, "bad request: %v\n", err)
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(req.DeadlineMS)*time.Millisecond)
	defer cancel()
	enc := json.NewEncoder(stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(run(ctx, req)); err != nil {
		fmt.Fprintf(stderr, "write response: %v\n", err)
		return 1
	}
	return 0
}

// recovered turns a panic into exit 1: Go's own exit code for one is 2, which
// the client reads as its own bad request. The stack, which names this build's
// source paths, is left out.
func recovered(stderr io.Writer, code *int) {
	if r := recover(); r != nil {
		fmt.Fprintf(stderr, "panic: %v\n", r)
		*code = 1
	}
}

// decode reads and checks a request. Messages never show the root.
func decode(r io.Reader) (request, error) {
	var req request
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		return req, fmt.Errorf("decode: %w", err)
	}
	if !filepath.IsAbs(req.Root) {
		return req, errors.New("root: not an absolute path")
	}
	if st, err := os.Stat(filepath.Join(req.Root, "go.mod")); err != nil || !st.Mode().IsRegular() {
		return req, errors.New("root: not a directory with a go.mod")
	}
	if req.DeadlineMS <= 0 {
		return req, errors.New("deadline_ms: not positive")
	}
	if len(req.Files) > maxFiles {
		return req, fmt.Errorf("files: %d, more than %d", len(req.Files), maxFiles)
	}
	for _, f := range req.Files {
		if !filepath.IsLocal(f.Path) || filepath.Clean(f.Path) != f.Path || !strings.HasSuffix(f.Path, ".go") {
			return req, fmt.Errorf("path %q: not a relative .go path", f.Path)
		}
	}
	return req, nil
}
