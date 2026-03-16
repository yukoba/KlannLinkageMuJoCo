import time

import mujoco
import mujoco.viewer

# Klannリンク機構のMuJoCo XML定義
# 簡略化された2Dの機構としてモデル化します。
KLANN_XML = """
<mujoco model="klann_linkage">
    <option gravity="0 0 -9.81" timestep="0.005" integrator="RK4"/>
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>

    <default>
        <joint type="hinge" axis="0 1 0" limited="false"/>
        <geom type="capsule" size="0.05" rgba="0.8 0.2 0.2 1" mass="1"/>
    </default>

    <asset>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
        <texture type="2d" name="grid" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    </asset>

    <worldbody>
        <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
        <geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1" material="grid"/>

        <!-- フレーム（ベース） -->
        <body name="frame" pos="0 0 1">
            <geom type="cylinder" size="0.1 0.1" rgba="0.2 0.8 0.2 1" euler="90 0 0" mass="10"/>
            <joint type="slide" axis="1 0 0" limited="true" range="-2 2"/>

            <!-- クランク -->
            <body name="crank" pos="0 0 0">
                <joint name="crank_joint" type="hinge"/>
                <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" rgba="0.8 0.8 0.2 1"/>

                <!-- コネクティングロッド -->
                <body name="connecting_rod" pos="0.2 0 0">
                    <joint name="crank_rod_joint" type="hinge"/>
                    <geom type="capsule" fromto="0 0 0 0.5 0 -0.2" size="0.02" rgba="0.2 0.8 0.8 1"/>

                    <!-- 脚 -->
                    <body name="leg" pos="0.5 0 -0.2">
                        <joint name="rod_leg_joint" type="hinge"/>
                        <geom type="capsule" fromto="0 0 0 -0.1 0 -0.8" size="0.02" rgba="0.8 0.2 0.8 1"/>
                    </body>
                </body>
            </body>

            <!-- 上部ロッカー -->
            <body name="upper_rocker" pos="-0.3 0 0.2">
                <joint name="upper_rocker_joint" type="hinge"/>
                <geom type="capsule" fromto="0 0 0 0.4 0 -0.1" size="0.02" rgba="0.2 0.2 0.8 1"/>
            </body>

            <!-- 下部ロッカー -->
            <body name="lower_rocker" pos="-0.4 0 -0.2">
                <joint name="lower_rocker_joint" type="hinge"/>
                <geom type="capsule" fromto="0 0 0 0.6 0 -0.1" size="0.02" rgba="0.8 0.5 0.2 1"/>
            </body>
        </body>
    </worldbody>

    <!-- クランクと他のリンクをつなぐ拘束 -->
    <equality>
        <connect body1="upper_rocker" body2="leg" anchor="0.4 0 -0.1"/>
        <connect body1="lower_rocker" body2="connecting_rod" anchor="0.6 0 -0.1"/>
    </equality>

    <actuator>
        <!-- クランクを回転させるモーター -->
        <motor joint="crank_joint" ctrlrange="-10 10" gear="5"/>
    </actuator>
</mujoco>
"""


class KlannLinkageSimulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(KLANN_XML, None)
        self.data = mujoco.MjData(self.model)

    def run(self, duration=10.0):
        """
        MuJoCoビューアを起動し、シミュレーションを実行します。
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()

            # クランクに一定のトルクを与えて回転させる
            self.data.ctrl[0] = 2.0

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
