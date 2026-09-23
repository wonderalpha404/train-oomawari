import unittest
from pathlib import Path

from oomawari import load_graph, longest_simple_path


class TestOomawari(unittest.TestCase):
    def test_small_graph_exact(self):
        data = Path(self.id().replace("/", "_") + ".csv")
        graph = {
            "A": [],
            "B": [],
            "C": [],
            "D": [],
        }
        from oomawari import Edge
        graph["A"] = [Edge("B", 1.0, "x"), Edge("C", 2.0, "y")]
        graph["B"] = [Edge("A", 1.0, "x"), Edge("C", 1.0, "x"), Edge("D", 10.0, "z")]
        graph["C"] = [Edge("A", 2.0, "y"), Edge("B", 1.0, "x"), Edge("D", 2.0, "y")]
        graph["D"] = [Edge("B", 10.0, "z"), Edge("C", 2.0, "y")]

        distance, edges = longest_simple_path("A", "D", graph)
        self.assertEqual(distance, 12.0)
        self.assertEqual([e[0] for e in edges] + [edges[-1][1]], ["A", "C", "B", "D"])


if __name__ == "__main__":
    unittest.main()
