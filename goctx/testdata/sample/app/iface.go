package app

import "sample/clock"

// Open reports whether now is within opening hours.
func Open(r clock.Ranger, now int) bool {
	return r.InRange(now, 9, 17)
}
