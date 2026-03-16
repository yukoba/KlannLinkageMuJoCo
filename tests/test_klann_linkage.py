import mujoco

from klann_linkage import KlannLinkageSimulation


def test_model_loads():
    model = mujoco.MjModel.from_xml_path("src/klann_linkage.xml", None)
    assert model is not None


def test_simulation_initializes():
    sim = KlannLinkageSimulation()
    assert sim.model is not None
    assert sim.data is not None
