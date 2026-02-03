from io import BytesIO
import os
# import matplotlib.pyplot as plt

# import caspian

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

# Provided Python utilities from tennlab framework/examples/common
from common.experiment import TennExperiment
import common.experiment
from common import env_tools as envt

from rss.gui import TennlabGUI
import rss.graphing as graphing

# typing:
from typing import override
from swarmsim.world.RectangularWorld import RectangularWorld

from common.argparse import ArgumentError


class ConnorMillingExperiment(TennExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    def __init__(self, args):
        super().__init__(args)
        self.world_yaml = args.world_yaml
        self.run_info = None

        self.track_history = args.track_history or args.log_trajectories
        self.log_trajectories = args.log_trajectories
        self.use_caspian = getattr(args, 'caspian', True)

        if self.agents is None and self.args.action != 'train':
            try:
                self.agents = self.p.experiment['agents']
            except (KeyError, IndexError, FileNotFoundError, AttributeError):
                pass

        # register controller type with RSS
        if self.use_caspian:
            from rss.CaspianBinaryController import CaspianBinaryController
            from rss.CaspianBinaryRemappedController import CaspianBinaryRemappedController
            self.controller, self.controller_remapped = CaspianBinaryController, CaspianBinaryRemappedController
        else:
            from rss.CasPyanBinaryController import CasPyanBinaryController
            from rss.CasPyanBinaryRemappedController import CasPyanBinaryRemappedController
            self.controller, self.controller_remapped = CasPyanBinaryController, CasPyanBinaryRemappedController

        self.n_inputs, self.n_outputs, _, _ = self.controller.get_default_encoders()

        self.start_paused = getattr(args, 'start_paused', False)

        self.log("initialized experiment_tenn2")

    def fetch_world_config(self):
        from swarmsim.world.RectangularWorld import RectangularWorldConfig
        from swarmsim import yaml
        if self.args.action != 'train':
            # try:
            #     with open(self.p.artifacts / 'env.yaml', 'r') as f:
            #         d = yaml.load(f)
            # except FileNotFoundError:
            #     pass
            # config = RectangularWorldConfig.from_dict(d)
            config = RectangularWorldConfig.from_yaml(self.world_yaml)
        else:
            config = RectangularWorldConfig.from_yaml(self.world_yaml)
        return config

    def simulate(self, processor, network, init_callback=None):
        from swarmsim.config import register_dictlike_type
        from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber as WorldSubscriber
        from swarmsim.world.simulate import main as simulator
        from swarmsim import metrics

        # setup network
        network.set_data("processor", self.processor_params)

        # register controller type with RSS
        if self.use_caspian:
            from rss.CaspianBinaryController import CaspianBinaryController
            from rss.CaspianBinaryRemappedController import CaspianBinaryRemappedController
            register_dictlike_type('controller', "CaspianBinaryController", CaspianBinaryController)
            register_dictlike_type('controller', "CaspianBinaryRemappedController", CaspianBinaryRemappedController)
        else:
            from rss.CasPyanBinaryController import CasPyanBinaryController
            from rss.CasPyanBinaryRemappedController import CasPyanBinaryRemappedController
            register_dictlike_type('controller', "CaspianBinaryController", CasPyanBinaryController)
            register_dictlike_type('controller', "CaspianBinaryRemappedController", CasPyanBinaryRemappedController)

        # setup world
        config = self.fetch_world_config()
        config.stop_at = self.cycles
        agent_config = config.spawners[0]['agent']
        agent_config['track_io'] = self.track_history
        controller_config = agent_config['controller']
        controller_config['neuro_track_all'] = self.viz
        controller_config['network'] = network
        if self.agents is not None:
            config.spawners[0]['n'] = self.agents

        config.metrics = [
            metrics.Circliness(history=max(self.cycles, 1), avg_history_max=450),
            # metrics.DelaunayDiffusion(history=max(self.cycles, 1)),
            # metrics.Aggregation(history=max(self.cycles, 1)),
            # metrics.DistanceSizeRatio(history=max(self.cycles, 1)),
        ]

        def callback(world, screen):
            a = world.selected
            if a and self.iostream:
                self.iostream.write_json({
                    "Neuron Alias": a.controller.neuron_ids,
                    "Event Counts": a.controller.neuron_counts
                })

        gui = TennlabGUI(x=0, y=0, h=0, w=300)
        gui.position = "sidebar_right"
        if self.viz is False or self.noviz:
            gui = False

        world_subscriber = WorldSubscriber(func=callback)

        simargs = dict(
            world_config=config,
            subscribers=[world_subscriber],
            gui=gui,
            show_gui=bool(gui),
            start_paused=self.start_paused,
        )

        # allow for callback to modify config
        if callable(init_callback):
            simargs = init_callback(self, simargs)

        world = simulator(**simargs)  # run simulator
        return world

    def extract_fitness(self, world_output: RectangularWorld):
        self.run_info = world_output.metrics[0].value_history if world_output.metrics else None
        if not world_output.metrics:
            return 0.0
        metric = world_output.metrics[0]
        return metric.average if metric.instantaneous else metric.value

    @override
    def fitness(self, processor, network, init_callback=None):
        world_final_state = self.simulate(processor, network, init_callback)
        return self.extract_fitness(world_final_state)

    def as_config_dict(self):
        d = super().as_config_dict()
        d.update({
            "world_yaml_path": self.world_yaml,
            # "run_info": self.run_info,
        })
        return d

    def save_network(self, net, path):
        if 'encoder_ticks' not in self.app_params:
            world = self.get_sample_world(delete_rss=False)
            self.app_params.update({'encoder_ticks': world.population[0].controller.neuro_tpc})
        super().save_network(net, path)

    def get_sample_world(self, delete_rss=True):
        import caspian
        from common.tennnetwork import make_template
        cycles = self.cycles
        self.cycles = 0
        proc = caspian.Processor(self.processor_params)
        template_net = make_template(proc, self.n_inputs, self.n_outputs, ...)
        world = self.simulate(proc, template_net)
        self.cycles = cycles
        if delete_rss:
            self.delete_rss()
        return world

    def delete_rss(self):
        if 'rss' in globals():
            del rss  # noqa

    def save_artifacts(self, evolver, *args, **kwargs):
        if super().save_artifacts(evolver, *args, **kwargs) is None:
            return
        self.p.save_yaml_artifact("env.yaml", self.get_sample_world(delete_rss=False))
        self.delete_rss()

    def get_env_info(self):
        d = super().get_env_info()
        try:
            swarmsim_path = envt.module_editable_path('swarmsim')
            d['.dependencies'].update({
                'swarmsim': {
                    'path': str(swarmsim_path.resolve()),  # pyright: ignore
                    "branch": envt.get_branch_name(swarmsim_path),  # pyright: ignore
                    "HEAD": envt.git_hash(swarmsim_path),  # pyright: ignore
                    "status": [s.strip() for s in envt.git_porcelain(swarmsim_path).split('\n')],  # pyright: ignore
                    'version': envt.get_module_version('swarmsim'),
                },
            })
        except Exception:
            d['.dependencies'].update({'swarmsim': envt.get_module_version('swarmsim')})
        return d


def run(app, args):

    # Set up simulator and network

    if args.stdin == "stdin":
        proc = None
        net = None
    else:
        # proc = caspian.Processor(app.processor_params)
        proc = None
        net = app.net

    # Run app and print fitness
    world = app.simulate(proc, net)
    fitness = app.extract_fitness(world)
    print(f"Fitness: {fitness:8.4f}")

    if args.log_trajectories:
        import matplotlib.pyplot as plt
        graphing.plot_multiple(world)
        graphing.plot_fitness(world)
        graphing.export(world, output_file=app.p.ensure_file_parents("agent_trajectories.xlsx"))
        if args.explore:
            app.p.explore()
        plt.show(block=True)
        # TODO: handle when no project
    else:
        if args.explore:
            app.p.explore()

    return fitness


def test(app, args):
    import caspian

    # Set up simulator and network
    proc = caspian.Processor(app.processor_params)
    net = app.net

    if args.positions:
        from rss.rss2 import PredefinedInitialization, SCALE
        import pandas as pd
        fpath = args.positions

        with open(fpath, 'rb') as f:
            xlsx = f.read()
        xlsx = pd.ExcelFile(BytesIO(xlsx))
        sheets = xlsx.sheet_names

        n_runs = len(sheets)

        pinit = PredefinedInitialization()  # num_agents isn't used yet here

        def setup_i(i):
            pinit.set_states_from_xlsx(args.positions, sheet_number=i)
            pinit.rescale(SCALE)

            def setup(world):
                world.init_type = pinit
                return world

            return setup

        # Run app and print fitness
        fitnesses = [app.fitness(proc, net, setup_i(i)) for i in tqdm(range(n_runs))]

        # print(f"Fitness: {fitness:8.8f}")
        print(fitnesses)
        return fitnesses
    else:
        raise ArgumentError(args.positions, "Positions not specified")


def get_parsers(parser, subpar):
    # this is a separate function so we can inherit options from this module
    sp = subpar.parsers

    for sub in sp.values():  # applies to everything
        sub.add_argument('-N', '--agents', default=None,  # override: use default from world.yaml
                         type=int, help="# of agents to run with.",)
        sub.add_argument('--world_yaml', default="rss/turbopi-milling/world.yaml",
                         type=str, help="path to yaml config for sim")

    # for key in ('test', 'run'):  # arguments that apply to test/validation and stdin
    #     pass  # sp[key].add_argument()

    # Training args
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")

    sp['run'].add_argument('--track_history', action='store_true',
                           help="pass this to enable sensor vs. output plotting.")
    sp['run'].add_argument('--log_trajectories', action='store_true',
                           help="pass this to log sensor vs. output to file.")
    sp['run'].add_argument('--start_paused', action='store_true',
                           help="pass this to pause the simulation at startup. Press Space to unpause.")
    sp['run'].add_argument('--caspian', action='store_true',
                           help="pass this to pause the simulation at startup. Press Space to unpause.")

    # Testing args
    sp['test'].add_argument('--positions', default=None,
                             help="file containing agent positions")
    sp['test'].add_argument('-p', '--processes', type=int, default=1,
                           help="number of threads for concurrent fitness evaluation.")

    return parser, subpar


def main():
    parser, subpar = common.experiment.get_parsers()
    parser, subpar = get_parsers(parser, subpar)  # modify parser

    args = parser.parse_args()

    args.environment = "connorsim_snn_eons-v01"  # type: ignore[reportAttributeAccessIssue]
    if args.project is None and args.logfile is None:
        args.logfile = "tenn2_train.log"

    app = ConnorMillingExperiment(args)

    # Do the appropriate action
    if args.action == "train":  # type: ignore[reportAttributeAccessIssue]
        common.experiment.train(app, args)
    elif args.action == "test":  # type: ignore[reportAttributeAccessIssue]
        test(app, args)
    elif args.action == "run":  # type: ignore[reportAttributeAccessIssue]
        run(app, args)
    else:
        raise RuntimeError("No action selected")


if __name__ == "__main__":
    main()
