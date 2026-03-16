import time

import mujoco
import mujoco.viewer


class KlannLinkageSimulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(
            "src/klann_linkage.xml",
            # "src/four_bar_linkage.xml",
            None)
        self.data = mujoco.MjData(self.model)

    def run(self, duration=100.0):
        """
        MuJoCoビューアを起動し、シミュレーションを実行します。
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()

            # クランクに一定のトルクを与えて回転させる
            self.data.ctrl[0] = 1.0

            while viewer.is_running() and time.time() - start_time < duration:
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # シミュレーションの進行を現実時間に合わせるための調整
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    sim = KlannLinkageSimulation()
    sim.run()
