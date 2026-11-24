# Security

## Reporting a vulnerability

Please report it privately through GitHub's private vulnerability reporting: the repository's **Security** tab,
then **Report a vulnerability**. Do not open a public issue for it.

## What metalscope does that matters

- **A hook into oMLX's Python.** `metalscope omlx install` writes a `.pth` file to the user site-packages of the
  Python 3.11 that oMLX.app bundles (`~/.local/lib/python3.11/site-packages`, shared by every standalone Python 3.11
  on the Mac). Its one line does nothing outside an oMLX.app bundle; inside one it puts metalscope on `sys.path` and
  imports it into the server. oMLX.app itself is not modified. `metalscope omlx uninstall` removes the file.
- **Local servers.** metalscope's routes run inside oMLX (`/scope` on its port) and in the ds4 proxy, which listens
  on 127.0.0.1 and starts ds4-server on 127.0.0.1. Both answer loopback clients only and refuse requests a browser
  sends for a page of another site (a non-loopback `Host` or `Origin`, or `Sec-Fetch-Site: cross-site`); on oMLX the
  same goes for a chat request that carries an `X-Scope-*` header. ds4-server itself listens on 127.0.0.1 port 8010
  behind the proxy and, unlike the proxy on 8009, does not refuse requests a browser sends for another site's page:
  a page that guesses the scope folder's path (`ds4/scope` under the data home) could make ds4-server create a new
  file inside that one folder (never one already there: a trace file is opened only if new). It cannot read traces.
- **Who may use them.** When oMLX has an API key, `/scope` asks for it too, checked by oMLX's own key check (so
  oMLX's setting that lets loopback clients skip the key applies here as well). Without a key in oMLX, and always on
  the ds4 proxy (ds4 has no key), any process on the Mac can read the live token stream, switch tracing and
  steering, and start or cancel jobs. On a Mac that other people log in to, set a key in oMLX and do not run the
  ds4 proxy. metalscope reads oMLX's key from `~/.omlx/settings.json` and sends it to oMLX's address only.
- **A patched ds4.** `metalscope ds4 install` clones antirez/ds4 at a pinned commit, applies the shipped patch and
  builds it locally.
- **Data on disk.** Traces hold the prompts and replies of the conversations they record, under the data home
  (`~/.metalscope` by default). metalscope creates that folder readable by its owner only (0700) and narrows it
  when it finds it broader.
- **Concept packs from Hugging Face.** `metalscope concept pull OWNER/REPO` downloads a repository and copies its
  texts, vectors (safetensors) and JSON into the concept folder. Nothing in a pack is executed. A vector of the
  wrong shape, a file that links outside the pack, and control characters in the text pull prints are refused. Pull
  only from publishers you trust: a vector changes what the model says.
