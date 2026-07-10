import json
import unittest

from src.communication.server import GestureServer


class FakeWebSocket:
    def __init__(self):
        self.sent_messages = []

    async def send(self, message):
        self.sent_messages.append(message)


class GestureServerRoomTests(unittest.IsolatedAsyncioTestCase):
    async def test_room_messages_are_forwarded_only_to_room_members(self):
        server = GestureServer()
        ws_one = FakeWebSocket()
        ws_two = FakeWebSocket()
        ws_three = FakeWebSocket()
        session_one = await server.register(ws_one)
        session_two = await server.register(ws_two)
        await server.register(ws_three)

        await server._handle_control_message(
            session_one,
            json.dumps({"type": "create_room", "role": "signer"}),
        )
        room_created = json.loads(ws_one.sent_messages[-1])
        room_code = room_created["room_code"]

        await server._handle_control_message(
            session_two,
            json.dumps(
                {
                    "type": "join_room",
                    "room_code": room_code,
                    "role": "receiver",
                }
            ),
        )

        ws_one.sent_messages.clear()
        ws_two.sent_messages.clear()
        ws_three.sent_messages.clear()

        await server._forward_message(session_one, "hello room")

        self.assertEqual(ws_two.sent_messages, ["hello room"])
        self.assertEqual(ws_three.sent_messages, [])

    async def test_leave_room_notifies_remaining_peers(self):
        server = GestureServer()
        ws_one = FakeWebSocket()
        ws_two = FakeWebSocket()
        session_one = await server.register(ws_one)
        session_two = await server.register(ws_two)

        await server._handle_control_message(
            session_one,
            json.dumps({"type": "create_room", "role": "signer"}),
        )
        room_code = json.loads(ws_one.sent_messages[-1])["room_code"]

        await server._handle_control_message(
            session_two,
            json.dumps(
                {
                    "type": "join_room",
                    "room_code": room_code,
                    "role": "receiver",
                }
            ),
        )

        ws_one.sent_messages.clear()
        await server._handle_control_message(
            session_two,
            json.dumps({"type": "leave_room"}),
        )

        self.assertTrue(ws_one.sent_messages)
        peer_left = json.loads(ws_one.sent_messages[-1])
        self.assertEqual(peer_left["type"], "peer_left")
        self.assertEqual(peer_left["room_code"], room_code)

    async def test_roomless_clients_keep_legacy_broadcast_behavior(self):
        server = GestureServer()
        ws_one = FakeWebSocket()
        ws_two = FakeWebSocket()
        session_one = await server.register(ws_one)
        await server.register(ws_two)

        await server._forward_message(session_one, "legacy broadcast")

        self.assertEqual(ws_two.sent_messages, ["legacy broadcast"])


if __name__ == "__main__":
    unittest.main()
