package notif

// Notifier sends messages. Defined in a different package from the handler that
// uses it, so it is out-of-diff context for a change to the handler.
type Notifier interface {
	Send(msg string) error
	Close() error
}
