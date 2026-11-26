package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestServe_BadRequestExits2(t *testing.T) {
	root := sampleRoot(t)
	noModule := t.TempDir()
	thirtyThree := strings.TrimSuffix(strings.Repeat(`{"path":"x.go"},`, 33), ",")
	tests := []struct {
		name, body, want string
	}{
		{"malformed", `{"root":`, "decode"},
		{"unknown field", fmt.Sprintf(`{"root":%q,"deadline_ms":1000,"files":[],"budget":1}`, root), "unknown field"},
		{"relative root", `{"root":"testdata/sample","deadline_ms":1000,"files":[]}`, "root: not an absolute path"},
		{"no go.mod", fmt.Sprintf(`{"root":%q,"deadline_ms":1000,"files":[]}`, noModule), "root: not a directory with a go.mod"},
		{"no deadline", fmt.Sprintf(`{"root":%q,"deadline_ms":0,"files":[]}`, root), "deadline_ms"},
		{"absolute path", fmt.Sprintf(`{"root":%q,"deadline_ms":1000,"files":[{"path":"/etc/x.go"}]}`, root), `"/etc/x.go"`},
		{"escaping path", fmt.Sprintf(`{"root":%q,"deadline_ms":1000,"files":[{"path":"../x.go"}]}`, root), `"../x.go"`},
		{"unclean path", fmt.Sprintf(`{"root":%q,"deadline_ms":1000,"files":[{"path":"./x.go"}]}`, root), `"./x.go"`},
		{"not Go", fmt.Sprintf(`{"root":%q,"deadline_ms":1000,"files":[{"path":"x.ts"}]}`, root), `"x.ts"`},
		{"33 files", fmt.Sprintf(`{"root":%q,"deadline_ms":1000,"files":[%s]}`, root, thirtyThree), "files: 33"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := serve(strings.NewReader(tt.body), &stdout, &stderr); code != 2 {
				t.Fatalf("exit %d, want 2", code)
			}
			if stdout.Len() != 0 || !strings.Contains(stderr.String(), tt.want) {
				t.Errorf("stdout %q, stderr %q: want %q", stdout.String(), stderr.String(), tt.want)
			}
			if tt.name != "relative root" && strings.Contains(stderr.String(), root) {
				t.Errorf("stderr shows the root: %q", stderr.String())
			}
		})
	}
}

func TestServe_ExpiredDeadlineExits0WithAnError(t *testing.T) {
	body := fmt.Sprintf(`{"root":%q,"deadline_ms":1,"files":[{"path":"handler.go","visible":[10],"changed":[10]}]}`, sampleRoot(t))
	var stdout, stderr bytes.Buffer
	if code := serve(strings.NewReader(body), &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, stderr.String())
	}
	var resp response
	if err := json.Unmarshal(stdout.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Files) != 1 || !strings.HasPrefix(resp.Files[0].Skipped, "load: ") || len(resp.Errors) == 0 {
		t.Errorf("want the file skipped with a load error, got %+v", resp)
	}
}

func TestServe_Response(t *testing.T) {
	body := fmt.Sprintf(`{"root":%q,"deadline_ms":60000,"files":[]}`, sampleRoot(t))
	var stdout, stderr bytes.Buffer
	if code := serve(strings.NewReader(body), &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, stderr.String())
	}
	// Empty lists, never null: the client's Vec fields need them.
	if got := strings.TrimSpace(stdout.String()); !strings.HasPrefix(got, `{"go":"go1.`) || !strings.HasSuffix(got, `,"errors":[],"files":[]}`) {
		t.Errorf("response %s", got)
	}
}

// Go exits 2 on a panic, the code for a bad request; recovered makes it 1 and
// keeps the stack, which names build paths, out of stderr.
func TestRecovered_PanicExits1(t *testing.T) {
	var stderr bytes.Buffer
	code := func() (code int) {
		defer recovered(&stderr, &code)
		var lines []string
		_ = lines[len(stderr.String())]
		return 0
	}()
	if got := stderr.String(); code != 1 || got != "panic: runtime error: index out of range [0] with length 0\n" {
		t.Errorf("exit %d, stderr %q", code, got)
	}
}

// GOMEMLIMIT is soft: past half again as much, the watchdog exits with exitMemory,
// which the client reports as a known failure.
func TestWatchHeap_ExitsPastTheHardCap(t *testing.T) {
	if got, want := hardCap(800<<20), uint64(1200<<20); got != want {
		t.Errorf("hardCap(800 MiB) = %d, want %d", got, want)
	}
	var stderr bytes.Buffer
	code := make(chan int, 1)
	go watchHeap(1, time.Millisecond, &stderr, func(c int) { code <- c })
	select {
	case got := <-code:
		if got != exitMemory || !strings.Contains(stderr.String(), "hard cap") {
			t.Errorf("exit %d, stderr %q", got, stderr.String())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no exit past the cap")
	}
}
