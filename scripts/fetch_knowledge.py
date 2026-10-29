"""Download GO and Reactome into the local knowledge cache (never committed). See gta.rag.fetch."""

import sys

from gta.rag.fetch import main

if __name__ == "__main__":
    sys.exit(main())
