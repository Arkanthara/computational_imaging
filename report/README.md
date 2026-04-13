# Report Build Notes

This report is configured to run Python fences through the `typst_pyexec` submodule
while reusing the `code/` uv environment.

## Preferred Command (Windows)

From repository root:

```powershell
.\report\run_typst_pyexec.ps1 build
```

Watch mode:

```powershell
.\report\run_typst_pyexec.ps1 watch -PreviewEngine typst
```

## Equivalent Raw uv Command

```bash
uv --project code run --with-editable report/typst_pyexec typst_pyexec build report/report.typ --typst-compile-arg --root --typst-compile-arg .
```

## Notes

- This workflow uses the `code/` environment (which pins `numpy==1.26.4`).
- No separate virtual environment is required inside `report/typst_pyexec`.
- The final Python blocks in `report/report.typ` are marked with `%| execute: false`
  to avoid long execution during quick build iteration.
