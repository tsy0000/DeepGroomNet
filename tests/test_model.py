import torch

from deepgroomnet.models import TAdaConvModel


def test_forward_shape_and_aux_head():
    model = TAdaConvModel(num_classes=5, use_cpam=True, use_stsa=True, drop_path=0.0)
    model.eval()
    x = torch.zeros(1, 3, 32, 224, 224)
    with torch.no_grad():
        logits, aux = model(x)
    assert logits.shape == (1, 5)
    assert aux.shape == (1, 5)


def test_ablations_drop_aux_head():
    for cpam, stsa in ((False, True), (True, False), (False, False)):
        model = TAdaConvModel(num_classes=5, use_cpam=cpam, use_stsa=stsa, drop_path=0.0)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 32, 224, 224))
        if cpam:
            assert isinstance(out, tuple) and out[1].shape == (1, 5)
        else:
            assert isinstance(out, tuple) and out[1] is None
        single = model(torch.zeros(1, 3, 32, 224, 224), return_aux=False)
        assert torch.is_tensor(single) and single.shape == (1, 5)


def test_stage_and_module_configuration():
    model = TAdaConvModel(num_classes=5)
    assert model.stsa3 is not None and model.stsa4 is not None
    assert len(model.stages) == 4
    assert model.head[-1].out_features == 5
