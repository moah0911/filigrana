# Worked example

`essay.md` is a 144-word article. Everything else here was produced by:

```console
inkmark key new
inkmark embed examples/essay.md --tier all --anti-slop \
    --out examples/essay.marked.md --sidecar examples/essay.sidecar.json
inkmark detect examples/essay.marked.md --tier all > examples/essay.detect.txt
inkmark audit  examples/essay.marked.md --tier all > examples/essay.audit.txt
```

Diff the two articles to see the cost: a handful of word choices move, and nothing a reader
would call an error. `essay.marked.md` vs `essay.md` is about 9 words out of 144.

These outputs are from a **throwaway key that is not committed**, so `detect` will not
reproduce them on your machine. Re-run the commands above with your own keyring instead. The
sidecar's `key_id` is likewise redacted; a real one names the key that minted it.

`essay.audit.txt` is the honest part: it prices what this article can support before you
publish it, and how fast that collapses under quoting and editing.
