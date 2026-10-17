"""
Pruebas del autómata con reloj simulado (no necesitan cámara ni red).
Ejecutar:  python test_automaton.py
"""
import unittest

from automaton import AutomatonConfig, TableState, TableStateMachine, classify


def make(**kw) -> TableStateMachine:
    cfg = AutomatonConfig(debounce_seconds=2, dirty_alert_seconds=10,
                          waiting_alert_seconds=20, stale_after_seconds=30)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return TableStateMachine(cfg)


def feed(m, personas, platos, t0, t1, step=0.5):
    """Alimenta la misma observación entre t0 y t1. Devuelve todos los eventos."""
    out, t = [], t0
    while t <= t1:
        out += m.update(personas, platos, now=t)
        t += step
    return out


class TruthTable(unittest.TestCase):
    def test_four_states(self):
        self.assertIs(classify(False, "ninguno"), TableState.VACIA)
        self.assertIs(classify(False, "vacios"), TableState.CON_RESTOS)
        self.assertIs(classify(False, "con_comida"), TableState.CON_RESTOS)
        self.assertIs(classify(True, "ninguno"), TableState.SENTADOS_SIN_COMIDA)
        self.assertIs(classify(True, "con_comida"), TableState.COMIENDO)
        self.assertIs(classify(True, "vacios"), TableState.COMIENDO)

    def test_invalid_platos(self):
        with self.assertRaises(ValueError):
            make().update(True, "otra_cosa", now=0)


class Behavior(unittest.TestCase):
    def test_first_observation_sets_state_immediately(self):
        m = make()
        m.update(True, "con_comida", now=0)
        self.assertIs(m.state, TableState.COMIENDO)

    def test_debounce_ignores_flicker(self):
        m = make()
        feed(m, False, "ninguno", 0, 5)
        m.update(True, "ninguno", now=5.5)          # 1 fotograma suelto
        feed(m, False, "ninguno", 6, 10)
        self.assertIs(m.state, TableState.VACIA)

    def test_cleaning_alert_fires_once_and_resolves(self):
        m = make()
        feed(m, False, "ninguno", 0, 3)
        ev = feed(m, False, "con_comida", 4, 30)
        alerts = [e for e in ev if e["type"] == "alert"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["kind"], "limpieza")
        self.assertIsNotNone(m.snapshot(30)["alert"])
        # el cronómetro cuenta desde que empezó la condición (t=4), no desde el debounce
        self.assertAlmostEqual(m.state_since, 4, places=1)
        ev = feed(m, False, "ninguno", 31, 36)       # alguien limpia
        self.assertTrue(any(e["type"] == "alert_resolved" for e in ev))
        self.assertIsNone(m.snapshot(36)["alert"])
        self.assertIs(m.state, TableState.VACIA)

    def test_waiting_alert_then_food_arrives(self):
        m = make()
        feed(m, True, "ninguno", 0, 19)
        self.assertIsNone(m.snapshot(19)["alert"])   # aún dentro del plazo
        ev = feed(m, True, "ninguno", 19.5, 25)
        self.assertEqual([e["kind"] for e in ev if e["type"] == "alert"], ["atencion"])
        ev = feed(m, True, "con_comida", 26, 30)     # llega la comida
        self.assertIs(m.state, TableState.COMIENDO)
        self.assertTrue(any(e["type"] == "alert_resolved" for e in ev))

    def test_eating_never_alerts(self):
        m = make()
        ev = feed(m, True, "con_comida", 0, 600, step=1)
        self.assertEqual([e for e in ev if e["type"] == "alert"], [])

    def test_stale_signal_resets_timers(self):
        m = make(stale_after_seconds=10)
        feed(m, False, "con_comida", 0, 5)
        m.update(False, "con_comida", now=500)       # la cámara estuvo caída 495 s
        self.assertIsNone(m.snapshot(500)["alert"])  # no hay alerta falsa
        self.assertAlmostEqual(m.snapshot(500)["seconds_in_state"], 0, places=1)

    def test_snapshot_countdown(self):
        m = make()
        feed(m, False, "con_comida", 0, 4)
        snap = m.snapshot(4)
        self.assertEqual(snap["state"], "con_restos")
        self.assertAlmostEqual(snap["seconds_to_alert"], 6, places=1)

    def test_configure_hot(self):
        m = make()
        cfg = m.configure(dirty_alert_seconds=300)
        self.assertEqual(cfg["dirty_alert_seconds"], 300)
        with self.assertRaises(ValueError):
            m.configure(dirty_alert_seconds=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
