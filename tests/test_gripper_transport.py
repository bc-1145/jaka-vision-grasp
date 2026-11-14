import json
import unittest
from unittest.mock import MagicMock, patch

from gripper_transport import CommunicationError, ControllerConnection


def packet(name="get_robot_state", **fields):
    return json.dumps(dict(cmdName=name, errorCode="0", **fields), ensure_ascii=False).encode()


class TransportTests(unittest.TestCase):
    def test_reuses_connection_and_accepts_split_json_with_braces_in_string(self):
        first = packet(msg='quoted } { " and 中文')
        split = first.index('中文'.encode()) + 1
        wire = MagicMock()
        wire.recv.side_effect = [first[:split], first[split:], packet()]
        with patch("gripper_transport.socket.create_connection", return_value=wire) as connect:
            client = ControllerConnection("test")
            self.assertEqual(client.request({"cmdName": "get_robot_state"})["msg"], 'quoted } { " and 中文')
            client.request({"cmdName": "get_robot_state"})
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(wire.sendall.call_count, 2)

    def test_closed_idle_connection_is_replaced_for_read_only(self):
        stale, fresh = MagicMock(), MagicMock()
        stale.recv.return_value = b""
        fresh.recv.return_value = packet()
        with patch("gripper_transport.socket.create_connection", side_effect=[stale, fresh]):
            client = ControllerConnection("test")
            self.assertEqual(client.request({"cmdName": "get_robot_state"})["errorCode"], "0")
        stale.close.assert_called_once()
        self.assertEqual(fresh.sendall.call_count, 1)

    def test_lost_tcp_position_read_is_retried_on_a_new_connection(self):
        stale, fresh = MagicMock(), MagicMock()
        stale.recv.return_value = b""
        fresh.recv.return_value = packet("get_actual_tcp_pos", position=[1, 2, 3, 4, 5, 6])
        with patch("gripper_transport.socket.create_connection", side_effect=[stale, fresh]):
            result = ControllerConnection("test").request({"cmdName": "get_actual_tcp_pos"})
        self.assertEqual(result["position"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(fresh.sendall.call_count, 1)

    def test_lost_write_reply_is_never_replayed(self):
        wire = MagicMock()
        wire.recv.side_effect = TimeoutError("reply lost after write")
        with patch("gripper_transport.socket.create_connection", return_value=wire) as connect:
            client = ControllerConnection("test")
            with self.assertRaises(CommunicationError):
                client.request({"cmdName": "set_analog_output", "type": 2, "index": 3, "value": 16800})
        self.assertEqual(wire.sendall.call_count, 1)
        self.assertEqual(connect.call_count, 1)
        wire.close.assert_called_once()
        self.assertIsNone(client.socket)

    def test_mismatched_or_multiple_replies_are_not_accepted_as_write_confirmation(self):
        for data in (packet(), packet("set_analog_output") + packet("set_analog_output")):
            with self.subTest(data=data):
                wire = MagicMock()
                wire.recv.return_value = data
                with patch("gripper_transport.socket.create_connection", return_value=wire):
                    with self.assertRaises(CommunicationError):
                        ControllerConnection("test").request({"cmdName": "set_analog_output"})
                self.assertEqual(wire.sendall.call_count, 1)

    def test_read_retry_is_bounded(self):
        wire = MagicMock()
        wire.recv.side_effect = TimeoutError("offline")
        with patch("gripper_transport.socket.create_connection", return_value=wire) as connect:
            with self.assertRaises(CommunicationError):
                ControllerConnection("test").request({"cmdName": "get_robot_state"})
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(wire.sendall.call_count, 2)
