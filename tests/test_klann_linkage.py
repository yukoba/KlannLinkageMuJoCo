import mujoco

from klann_linkage import KLANN_XML, KlannLinkageSimulation


def test_model_loads():
    model = mujoco.MjModel.from_xml_string(KLANN_XML, None)
    assert model is not None


def test_simulation_initializes():
    sim = KlannLinkageSimulation()
    assert sim.model is not None
    assert sim.data is not None
