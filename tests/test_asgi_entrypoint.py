import unittest

from asgi import app


class AsgiEntrypointTests(unittest.TestCase):
    def test_http_routes_are_exposed(self):
        route_methods = {}
        for route in app.routes:
            methods = getattr(route, "methods", None)
            if methods is None:
                continue
            route_methods.setdefault(route.path, set()).update(methods)

        self.assertIn("/", route_methods)
        self.assertIn("GET", route_methods["/"])
        self.assertIn("HEAD", route_methods["/"])

        self.assertIn("/health", route_methods)
        self.assertIn("GET", route_methods["/health"])
        self.assertIn("HEAD", route_methods["/health"])

    def test_websocket_route_is_exposed(self):
        websocket_paths = {
            route.path
            for route in app.routes
            if getattr(route, "path", None) and route.__class__.__name__ == "APIWebSocketRoute"
        }
        self.assertIn("/ws", websocket_paths)


if __name__ == "__main__":
    unittest.main()
