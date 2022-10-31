from time import process_time

from torch_utils import training_stats

from . import util


class Tracker:
    """
    utility class for tracking different iterations during training
    """

    def __init__(
        self,
        kimg_per_tick: int,
        total_kimg: int,
        img_snapshot_ticks: int = 50,
        network_snapshot_ticks: int = 50,
    ):

        # attributes related to the number of images processed so far
        self.kimg_per_tick = kimg_per_tick
        self.total_kimg = total_kimg
        self.cur_nimg = 0
        self.batch_idx = 0
        self.cur_tick = 0
        self.tick_start_nimg = 0

        # timing attributes
        self.tick_start_time = 0
        self.tick_end_time = 0
        self.start_time = process_time()
        self.maintenance_time = 0

        # how frequently save image snapshots and network pickles
        self.img_snapshot_ticks = img_snapshot_ticks
        self.network_snapshot_ticks = network_snapshot_ticks

    def start_ticks(self):
        self.tick_start_time = process_time()
        self.maintenance_time = self.tick_start_time - self.start_time

    def step(self, n_img):
        self.cur_nimg += n_img
        self.batch_idx += 1

    def is_done(self):
        return self.cur_nimg >= (self.total_kimg * 1000)

    def should_report(self):
        return (
            (self.is_done())
            or (self.cur_tick == 0)
            or (self.cur_nimg >= self.tick_start_nimg + self.kimg_per_tick * 1000)
        )

    def should_save_network_snapshot(self):
        return (self.network_snapshot_ticks is not None) and (
            self.is_done() or self.cur_tick % self.network_snapshot_ticks == 0
        )

    def should_save_image_snapshot(self):
        return (self.img_snapshot_ticks is not None) and (
            self.is_done() or self.cur_tick % self.img_snapshot_ticks == 0
        )

    def report_time_stats(self):
        """
        report the timing statistics at the end of the tick
        """
        self.tick_end_time = process_time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', self.cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', self.cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {util.format_time(training_stats.report0('Timing/total_sec', self.tick_end_time - self.start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', self.tick_end_time - self.tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (self.tick_end_time - self.tick_start_time) / (self.cur_nimg - self.tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', self.maintenance_time):<6.1f}"]
        print(" ".join(fields))

    def update_states(self):
        """
        update the internal states for the next tick
        """
        self.cur_tick += 1
        self.tick_start_nimg = self.cur_nimg
        self.tick_start_time = process_time()
        self.maintenance_time = self.tick_start_time - self.tick_end_time

    def reset(self):
        self.cur_nimg = 0
        self.batch_idx = 0
        self.cur_tick = 0
        self.tick_start_nimg = 0

        self.tick_start_time = 0
        self.tick_end_time = 0
        self.start_time = process_time()
        self.maintenance_time = 0
