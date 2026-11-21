import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_provider.data_loader import Dataset_Custom, fit_daily_cycle
from model.FAiT import Model


class DailyTemplateTest(unittest.TestCase):
    def test_fit_only_uses_training_rows_and_window_hours_follow_timestamps(self):
        dates = pd.date_range("2024-01-01 21:45", periods=960, freq="15min")
        frame = pd.DataFrame({"date": dates, "OT": dates.hour * 2.0 + dates.day})
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "example.csv"
            frame.to_csv(csv, index=False)
            kwargs = dict(root_path=folder, data_path=csv.name, size=[96, 48, 96],
                          features="S", scale=False, daily_template=True)
            train = Dataset_Custom(flag="train", **kwargs)
            cycle = fit_daily_cycle(train)
            stop = int(len(frame) * 0.7)
            expected = frame.iloc[:stop].groupby(dates.hour[:stop])["OT"].mean()
            np.testing.assert_allclose(cycle[:, 0], expected - expected.mean(), atol=1e-5)
            frame.loc[stop:, "OT"] += 10000
            frame.to_csv(csv, index=False)
            np.testing.assert_array_equal(fit_daily_cycle(Dataset_Custom(flag="train", **kwargs)), cycle)
            valid = Dataset_Custom(flag="valid", **kwargs)
            with self.assertRaises(ValueError):
                fit_daily_cycle(valid)
            _, _, history, future = valid[0]
            np.testing.assert_array_equal(history[:, 0], dates[stop-96:stop].hour)
            np.testing.assert_array_equal(future[-96:, 0], dates[stop:stop+96].hour)

    def test_daily_predicts_residual_and_restores_future_profile(self):
        cfg = SimpleNamespace(seq_len=96, pred_len=192, enc_in=2, d_model=16, d_ff=32,
                              embed_size=8, factor=3, n_heads=4, e_layers=1,
                              output_attention=False, activation="gelu", CKA_flag=0, dropout=0.)
        torch.manual_seed(42)
        baseline = Model(cfg).eval()
        cfg.daily_template = 1
        daily = Model(cfg).eval()
        state = baseline.state_dict()
        self.assertNotIn("daily_cycle", state)
        profile = torch.arange(24).float() - 11.5
        state["daily_cycle"] = torch.stack((profile, -profile), dim=1)
        daily.load_state_dict(state, strict=True)
        x = torch.randn(2, 96, 2)
        hours = (torch.tensor([17, 23])[:, None] + torch.arange(288)) % 24
        with torch.inference_mode():
            expected = baseline(x - daily.daily_cycle[hours[:, :96]]) + daily.daily_cycle[hours[:, 96:]]
            # A 13-step decoder prefix puts its first mark at a different hour than the first target.
            actual = daily(x, hours[:, :96, None], None, hours[:, 83:, None])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
