import unittest

from oomawari import Edge, longest_simple_path


class TestOomawari(unittest.TestCase):
    def test_small_graph_exact(self):
        stations = ["A", "B", "C", "D"]
        index = {name: i for i, name in enumerate(stations)}
        edges = [
            Edge("A", "B", 1.0, "x"),
            Edge("A", "C", 2.0, "y"),
            Edge("B", "C", 1.0, "x"),
            Edge("B", "D", 10.0, "z"),
            Edge("C", "D", 2.0, "y"),
        ]

        distance, route = longest_simple_path("A", "D", stations, edges, index)

        self.assertAlmostEqual(distance, 13.0)
        self.assertEqual([route[0][0]] + [e[1] for e in route], ["A", "C", "B", "D"])


if __name__ == "__main__":
    unittest.main()
