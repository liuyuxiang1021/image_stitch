"""Keep ``python -m image_stitch`` available for existing commands."""

from point_line_stitch.cli import main

raise SystemExit(main())
