import asyncio
import unittest

from src.web.app import DualTalkSession


class DualTalkSessionStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = DualTalkSession(
            asyncio.get_running_loop(),
            "ws://127.0.0.1:8765/ws",
        )

    async def asyncTearDown(self):
        await self.session.stop()

    async def test_join_room_stays_pending_until_server_confirms(self):
        await self.session.join_room("dt-4821", role="signer")

        snapshot = self.session.build_snapshot()

        self.assertEqual(snapshot["stage"], "room")
        self.assertEqual(snapshot["room_code"], "DT-4821")
        self.assertFalse(snapshot["status"]["connected"])
        self.assertIn("Joining", snapshot["status"]["text"])
        self.assertIsNone(self.session.get_room_code())
        self.assertEqual(self.session.get_resume_room_code(), "DT-4821")

    async def test_room_confirmation_promotes_pending_room(self):
        await self.session.join_room("DT-4821", role="receiver")
        self.session._set_server_connected(True)
        self.session._apply_room_join("DT-4821", participants=[])

        snapshot = self.session.build_snapshot()

        self.assertEqual(self.session.get_room_code(), "DT-4821")
        self.assertEqual(self.session.get_resume_room_code(), "DT-4821")
        self.assertTrue(snapshot["status"]["connected"])
        self.assertEqual(snapshot["role"], "receiver")


if __name__ == "__main__":
    unittest.main()
