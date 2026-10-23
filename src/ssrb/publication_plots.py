"""Render the publication layouts from explicitly selected aggregate results."""
from pathlib import Path
import json
from .data import sha256,write_json

def run(profile,source,output):
    source,output=Path(source),Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.suffix.lower()!='.pdf':
        raise ValueError('publication plots require a PDF output path')
    before=sha256(source)
    if profile=='representation':
        from .paper_representation_plot import main
    elif profile=='budget':
        from .paper_budget_plot import main
    elif profile=='probunet':
        from .paper_probunet_plot import main
    elif profile=='mechanism':
        from .paper_mechanism_plot import main
    elif profile=='qualitative':
        from .paper_qualitative_plot import main
    elif profile=='bsds-construction':
        from .paper_bsds_plot import main
    elif profile=='assignment-examples':
        from .paper_assignment_plot import main
    else:
        raise ValueError('unsupported publication plot profile')
    main(source,output)
    if sha256(source)!=before:
        raise ValueError('plot input changed')
    record_path=output.with_suffix('.record.json')
    record=json.loads(record_path.read_text()) if record_path.exists() else {}
    record.update(profile=profile,input_sha256=before,output_sha256=sha256(output),numerical_source='caller-supplied results; inspect input provenance before interpreting the figure')
    write_json(record_path,record)
    return str(output)
