package tick

// Run ticks three times: the second use is inside the first one's site.
func Run() int {
	a := Tick(1)
	a++
	a++
	b := Tick(a)
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	b++
	return Tick(b)
}
