#!/bin/bash
# Node tests for the extension's pure logic (Pi itself loads it with jiti): bundle each test/*.test.ts with
# esbuild (node does not run this TypeScript as is), then `node --test`.
#   npm install          (once: esbuild and typescript, the devDependencies)
#   ./test.sh            (or: npm test)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# Pi's packages: a managed install (~/.pi/agent/install, `pi update`), the Homebrew/npm global one, else this
# folder's own (npm install here).
default_modules=/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/node_modules
if [ -r "$HOME/.pi/agent/install/current-version" ]; then
	v=$(cat "$HOME/.pi/agent/install/current-version")
	[ -d "$HOME/.pi/agent/install/releases/$v/node_modules" ] && default_modules=$HOME/.pi/agent/install/releases/$v/node_modules
fi
[ -d "$default_modules/@earendil-works" ] || default_modules=$here/node_modules
pi_modules=${PI_MODULES:-$default_modules}
if [ ! -d "$pi_modules/@earendil-works" ]; then
	echo "no Pi packages at $pi_modules: run \`cd src/metalscope/frontends/pi && npm install\`, or set PI_MODULES to Pi's node_modules" >&2
	exit 1
fi
esbuild=${ESBUILD:-$here/node_modules/.bin/esbuild}
if [ ! -x "$esbuild" ]; then
	echo "no esbuild: run \`cd src/metalscope/frontends/pi && npm install\` (installs esbuild and typescript)" >&2
	exit 1
fi
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
# Type-check first (esbuild strips types without checking them): tsconfig.json resolves Pi's packages
# through the .pi-modules link, and typescript is a devDependency.
tsc=${TSC:-$here/node_modules/.bin/tsc}
if [ ! -x "$tsc" ]; then
	echo "no tsc at $tsc: run \`cd src/metalscope/frontends/pi && npm install\` (installs esbuild and typescript)" >&2
	exit 1
fi
ln -sfn "$pi_modules" "$here/.pi-modules"
(cd "$here" && "$tsc" -p tsconfig.json) || { echo "type-check failed" >&2; exit 1; }
# Pi's own packages stay external; the bundles resolve them from Pi's install.
mkdir "$out/node_modules"
ln -s "$pi_modules/@earendil-works" "$out/node_modules/@earendil-works"
for t in "$here"/test/*.test.ts; do
	"$esbuild" "$t" --bundle --platform=node --format=esm '--external:@earendil-works/*' --log-level=warning \
		--outfile="$out/$(basename "${t%.ts}").mjs"
done
# A home of their own: the extension reads oMLX's settings (its key) under it.
HOME="$out" node --test "$out"/*.test.mjs
