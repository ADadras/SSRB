from importlib import import_module


def test_publication_layouts_load_with_their_declared_specs():
    for name in ('representation','budget','probunet','mechanism','qualitative','bsds','assignment'):
        module=import_module('ssrb.paper_'+name+'_plot')
        assert callable(module.main)
    from ssrb.paper_qualitative_plot import SPECS
    assert len(SPECS)==4
    assert SPECS[0].visible_crop
