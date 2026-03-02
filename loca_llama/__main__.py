"""Allow running as `python -m loca_llama`."""

import sys

# If no args (or just --interactive), launch interactive mode
if len(sys.argv) <= 1 or sys.argv[1] == "--interactive":
    from .interactive import main_interactive
    main_interactive()
else:
    from .cli import main
    main()
