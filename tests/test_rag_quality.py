import unittest

from df_kefu_baseline.answer import AnswerEngine, readable_chunk_text
from df_kefu_baseline.manuals import load_manuals
from df_kefu_baseline.query_planner import build_query_plan


class RagQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AnswerEngine()

    def test_english_bundle_is_split_into_product_documents(self):
        manuals = load_manuals()
        english_docs = [manual.name for manual in manuals if manual.name.startswith("汇总英文手册::")]

        self.assertGreaterEqual(len(english_docs), 15)
        self.assertTrue(any("vacuum" in name for name in english_docs))
        self.assertTrue(any("landline" in name for name in english_docs))
        self.assertTrue(any("earphones" in name for name in english_docs))
        self.assertTrue(any("snowmobile" in name for name in english_docs))

    def test_vacuum_query_does_not_retrieve_tv_or_microwave(self):
        question = (
            "What are the two primary modes available on a vacuum cleaner that can "
            "be used to tailor its performance to meet different home cleaning needs?"
        )
        plan = build_query_plan(question, self.engine.manual_names)
        results = self.engine.retrieve(plan)
        combined = "\n".join(readable_chunk_text(result.chunk) for result in results[:3]).lower()

        self.assertIn("vacuum", combined)
        self.assertNotIn("key lock", combined)
        self.assertNotIn("microwave oven", combined)

    def test_landline_query_does_not_retrieve_motherboard_connectors(self):
        question = "How can you connect the base station of a landline?"
        plan = build_query_plan(question, self.engine.manual_names)
        results = self.engine.retrieve(plan)
        combined = "\n".join(readable_chunk_text(result.chunk) for result in results[:3]).lower()

        self.assertIn("base station", combined)
        self.assertNotIn("rog_ext", combined)
        self.assertNotIn("usb 3.0 module", combined)

    def test_landline_overview_and_status_behavior_routes_are_distinct(self):
        overview = self.engine.answer("What is the overview of the base station of a landline?", qid=351)
        led_status = self.engine.answer(
            "How can you know about the current status with different LED indicator behavior on a landline?",
            qid=356,
        )

        self.assertIn("Overview of the base station", overview)
        self.assertNotIn("DC input jack", overview)
        self.assertIn("Behavior of the LED indicator", led_status)
        self.assertIn("current status", led_status)
        self.assertNotIn("Loudspeaker Battery door", led_status)

    def test_earphones_components_query_does_not_retrieve_landline_troubleshooting(self):
        question = "After the earphones is in my hand, what are the components I should have?"
        answer = self.engine.answer(question, qid=296).lower()

        self.assertIn("earbud", answer)
        self.assertIn("charging", answer)
        self.assertNotIn("handset", answer)
        self.assertNotIn("base station", answer)

    def test_oven_accessory_queries_return_distinct_focused_answers(self):
        questions = {
            "225": "如何使用烤箱的烤架烤盘套装？",
            "226": "如何使用烤箱的油脂过滤器？",
            "227": "如何使用烤箱的滑动搁架？",
        }
        answers = {qid: self.engine.answer(question, qid=qid) for qid, question in questions.items()}

        self.assertIn("烤架烤盘套装", answers["225"])
        self.assertIn("油脂过滤器", answers["226"])
        self.assertIn("滑动搁架", answers["227"])
        for answer in answers.values():
            self.assertNotIn("首次使用烤箱", answer)
        self.assertEqual(len(set(answers.values())), len(answers))

    def test_generator_switch_overview_avoids_front_matter(self):
        answer = self.engine.answer("我注意到发电机上有两种不同的开关，请为我介绍一下。", qid=160)

        self.assertIn("发动机开关", answer)
        self.assertIn("经济控制开关", answer)
        self.assertNotIn("保修", answer)

    def test_generator_shutdown_query_prefers_shutdown_procedure(self):
        answer = self.engine.answer("我需要让发电机的发动机停机，告诉我该如何操作。", qid=166)

        self.assertIn("断开所有电气设备", answer)
        self.assertIn("燃油开关旋钮", answer)
        self.assertNotIn("保修", answer)

    def test_generator_start_steps_do_not_route_to_cannot_start(self):
        answer = self.engine.answer("启动发电机发动机的前两个步骤是什么？", qid=158)

        self.assertIn("启动发动机前", answer)
        self.assertIn("通气旋钮", answer)
        self.assertNotIn("发动机无法启动", answer)

    def test_blower_focus_queries_return_distinct_blocks(self):
        answers = {
            "66": self.engine.answer("使用吹风机时，如何调节化油器？", qid=66),
            "67": self.engine.answer("吹风机冷机时，该如何启动？", qid=67),
            "68": self.engine.answer("吹风机热机时，该如何启动？", qid=68),
        }

        self.assertIn("化油器", answers["66"])
        self.assertIn("冷机启动", answers["67"])
        self.assertIn("热机启动", answers["68"])
        self.assertEqual(len(set(answers.values())), len(answers))

    def test_fitness_tracker_problem_and_lock_queries_do_not_return_payment(self):
        problem_answer = self.engine.answer("使用健身追踪器时，我可能会遇到哪些问题以及该如何解决？", qid=142)
        lock_answer = self.engine.answer("我想给健身追踪器设置锁屏，该如何实现？", qid=144)

        self.assertIn("其他问题", problem_answer)
        self.assertIn("重启", problem_answer)
        self.assertIn("设备锁", lock_answer)
        self.assertIn("PIN", lock_answer)
        self.assertNotIn("进行消费", lock_answer)

    def test_processor_unit_component_query_prefers_vr_over_generic_parts(self):
        answer = self.engine.answer("设备或系统中构成处理器单元的关键组件或部件有哪些？", qid=195)

        self.assertIn("处理器单元", answer)
        self.assertIn("状态指示灯", answer)
        self.assertIn("HDMI输出端口", answer)
        self.assertIn("DC IN 12V接口", answer)
        self.assertNotIn("洗碗机", answer)
        self.assertNotIn("喷淋臂", answer)

    def test_camera_queries_avoid_flash_toc_duplicates(self):
        answers = {
            "283": self.engine.answer("How do you mount the lens of a camera when preparing for photography?", qid=283),
            "284": self.engine.answer("How do you install the card into a camera before photography?", qid=284),
            "288": self.engine.answer("Before taking photos, we need to know How to use the eyepiece cover of the camera?", qid=288),
            "292": self.engine.answer("How can I delete a single image from my camera if I don't like the photo?", qid=292),
            "294": self.engine.answer("How do I print photos using the camera's CP direct method after taking them?", qid=294),
        }

        self.assertIn("Remove the caps", answers["283"])
        self.assertIn("CF card", answers["284"])
        self.assertIn("eyepiece cover", answers["288"].lower())
        self.assertIn("Erasing a Single Image", answers["292"])
        self.assertIn("CP Direct", answers["294"])
        for answer in answers.values():
            self.assertNotIn("Flash Photography 91", answer)
            self.assertNotIn("C.Fn-14", answer)

    def test_component_overview_and_action_queries_are_separated(self):
        lid = self.engine.answer("How can you set pressure cooking lid for a multi-use pressure cooker and air fryer?", qid=388)
        anti_overview = self.engine.answer(
            "What do you know about anti-block shield for a multi-use pressure cooker and air fryer?",
            qid=390,
        )
        anti_set = self.engine.answer("How can you set anti-block shield for a multi-use pressure cooker and air fryer?", qid=392)
        ring_overview = self.engine.answer("What do you know about sealing ring of a multi-use pressure cooker and air fryer?", qid=397)

        self.assertIn("Pressure cooking lid", lid)
        self.assertIn("Anti-block shield", anti_overview)
        self.assertIn("prevents food particles", anti_overview)
        self.assertIn("Install the anti-block shield", anti_set)
        self.assertIn("Sealing ring", ring_overview)
        self.assertNotIn("Remove the sealing ring", ring_overview)

    def test_float_valve_and_silicone_cap_queries_are_focused(self):
        valve = self.engine.answer("How can you set float valve for a multi-use pressure cooker and air fryer?", qid=389)
        cap = self.engine.answer("How can you set silicone cap and float valve for a multi-use pressure cooker and air fryer?", qid=393)

        self.assertIn("Drop the narrow end", valve)
        self.assertIn("silicone cap", cap.lower())
        self.assertIn("bottom", cap.lower())
        self.assertNotEqual(valve, cap)

    def test_title_only_evidence_expands_to_substantive_answer(self):
        connect = self.engine.answer("How can you connect the base station of a landline?", qid=352)
        throttle = self.engine.answer("What are the steps to adjust the throttle cable on a snowmobile?", qid=419)

        self.assertIn("DC input jack", connect)
        self.assertNotEqual(connect.strip(), "The relevant instructions are:\n1. Connect the base station")
        self.assertIn("Loosen the adjuster lock nut", throttle)
        self.assertNotEqual(throttle.strip(), "The relevant instructions are:\n1. THROTTLE CABLE ADJUSTMENT")

    def test_vacuum_troubleshooting_extracts_troubleshooting_block(self):
        answer = self.engine.answer(
            "What are the troubleshooting steps if the vacuum indicates a problem?",
            qid=412,
        )

        self.assertIn("troubleshooting indicator", answer.lower())
        self.assertNotIn("CLEANING THE EXTRACTORS", answer)


if __name__ == "__main__":
    unittest.main()
