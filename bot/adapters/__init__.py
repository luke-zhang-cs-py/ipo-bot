"""One adapter per source. Each is a module with NAME, TERMS (on what terms the source is read), url() and
parse(); base.read() fetches, parses and checks the shape of what came back, and base.first_ok() tries a
source's fallbacks in order."""
