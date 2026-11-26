package sample

import "sample/notif"

// Handler holds a Notifier whose definition lives in another package.
type Handler struct {
	notifier notif.Notifier
}

func NewHandler(notifier notif.Notifier) *Handler {
	return &Handler{notifier: notifier}
}

func (h *Handler) Send(msg string) error {
	return h.notifier.Send(msg)
}
