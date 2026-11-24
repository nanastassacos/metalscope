"""The oMLX backend: patches that connect metalscope to a running oMLX (patches, applied by boot at import, and
patches_mtp, applied when an MTP model attaches), the /scope routes' view of it (backend), its admin API (admin), its
install hook (install) and its loader (load, ``--loader omlx``). Imports nothing: boot runs in every process of oMLX's
Python."""
