import unittest

from df_kefu_baseline.data import normalize_question, split_question_parts
from df_kefu_baseline.policy import answer_policy_subquestion
from df_kefu_baseline.submission_format import sanitize_submission_text
from scripts.generate_submission import format_competition_ret
from scripts.audit_submission import parse_ret


class MultiQuestionFormatTests(unittest.TestCase):
    def test_normalizes_quoted_question_groups(self):
        raw = '"请问你们家的商品支持7天无理由退换货吗？",\n"需要自己承担运费吗？"'

        self.assertEqual(
            split_question_parts(raw),
            ["请问你们家的商品支持7天无理由退换货吗？", "需要自己承担运费吗？"],
        )

    def test_strips_single_trailing_quote_comma_without_splitting(self):
        raw = '"复杂售后问题该怎么处理？",'

        self.assertEqual(normalize_question(raw), "复杂售后问题该怎么处理？")
        self.assertEqual(split_question_parts(raw), ["复杂售后问题该怎么处理？"])

    def test_formats_multiple_answers_as_official_json_strings(self):
        ret = format_competition_ret(["答案一。", "答案二。"])

        self.assertEqual(ret, '"答案一。","答案二。"')
        answer, images, error = parse_ret(ret)
        self.assertEqual(error, "")
        self.assertEqual(answer, "答案一。\n答案二。")
        self.assertEqual(images, [])

    def test_formats_image_list_like_official_csv_sample(self):
        ret = format_competition_ret("说明<PIC><PIC>\nRelated images: Manual14_33, Manual14_34")

        self.assertEqual(ret, '"说明<PIC><PIC>",["Manual14_33", "Manual14_34"]')
        answer, images, error = parse_ret(ret)
        self.assertEqual(error, "")
        self.assertEqual(answer, "说明<PIC><PIC>")
        self.assertEqual(images, ["Manual14_33", "Manual14_34"])

    def test_sanitizes_hidden_and_emoji_characters(self):
        self.assertEqual(sanitize_submission_text("答\u200b案🙂“好”"), '答案"好"')

    def test_policy_subanswer_keeps_context_for_ambiguous_shipping_fee(self):
        context = "请问你们家的商品支持7天无理由退换货吗？\n需要自己承担运费吗？"

        answer = answer_policy_subquestion("需要自己承担运费吗？", context)

        self.assertIn("非质量问题退换货", answer)
        self.assertIn("买家承担", answer)


if __name__ == "__main__":
    unittest.main()
