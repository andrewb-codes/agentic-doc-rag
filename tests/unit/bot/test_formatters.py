from agentic_rag.bot.client import (
    BotAskResponse,
    BotDocumentChunkResponse,
    BotUnsupportedClaim,
    BotVerificationResult,
)
from agentic_rag.bot.formatters import (
    format_answer_message,
    format_sources_message,
    format_verification_details,
    split_telegram_text,
)
from agentic_rag.core.enums import AnswerStatus, VerificationVerdict


def ask_response(
    *,
    answer: str = "Срок подачи отчёта — 31 июля.",
    answer_status: AnswerStatus = AnswerStatus.ANSWERED,
    verdict: VerificationVerdict | None = VerificationVerdict.SUPPORTED,
    unsupported_claims: list[BotUnsupportedClaim] | None = None,
    missing_information: list[str] | None = None,
    confidence: float | None = 1.0,
) -> BotAskResponse:
    return BotAskResponse(
        answer=answer,
        answer_status=answer_status,
        chunks=[
            BotDocumentChunkResponse(
                id=1,
                document_id=10,
                page=2,
                chunk_index=0,
                text="Отчёт нужно подать не позднее 31 июля текущего года.",
                source="policy.pdf",
            ),
            BotDocumentChunkResponse(
                id=2,
                document_id=10,
                page=2,
                chunk_index=1,
                text="Дублирующий фрагмент с той же страницы.",
                source="policy.pdf",
            ),
            BotDocumentChunkResponse(
                id=3,
                document_id=11,
                page=5,
                chunk_index=0,
                text="Форма отчёта описана в приложении.",
                source="appendix.pdf",
            ),
        ],
        verification_result=BotVerificationResult(
            verdict=verdict,
            unsupported_claims=unsupported_claims or [],
            missing_information=missing_information or [],
            confidence=confidence,
        ),
    )


def test_format_answer_message_shows_short_supported_status_without_confidence() -> None:
    text = format_answer_message(
        ask_response(
            missing_information=["Не хватает приложения 2."],
            confidence=0.73,
        )
    )

    assert "💬 Ответ" in text
    assert "🔎 Где искали" in text
    assert "• policy.pdf — стр. 2" in text
    assert "• appendix.pdf — стр. 5" in text
    assert "✅ Ответ подтверждён документацией" in text
    assert "0.73" not in text
    assert "73%" not in text
    assert "Не хватает приложения 2." not in text


def test_format_answer_message_shows_unsupported_claims_without_missing_info() -> None:
    text = format_answer_message(
        ask_response(
            verdict=VerificationVerdict.UNSUPPORTED,
            unsupported_claims=[
                BotUnsupportedClaim(
                    claim="В документе нет штрафа 10%.",
                    reason="Размер штрафа не указан.",
                )
            ],
            missing_information=["Нет финансового приложения."],
            confidence=0.22,
        )
    )

    assert "⚠️ Не все утверждения удалось подтвердить по документам." in text
    assert "Не подтверждено:" in text
    assert "• В документе нет штрафа 10%." in text
    assert "Размер штрафа не указан." not in text
    assert "Нет финансового приложения." not in text
    assert "0.22" not in text
    assert "22%" not in text


def test_format_answer_message_shows_neutral_status_for_missing_verdict() -> None:
    text = format_answer_message(ask_response(verdict=None))

    assert "ℹ️ Ответ не был проверен." in text


def test_format_answer_message_does_not_duplicate_not_found_status() -> None:
    text = format_answer_message(
        ask_response(
            answer="Ответ на вопрос не найден в документах.",
            answer_status=AnswerStatus.NOT_FOUND,
            verdict=VerificationVerdict.SUPPORTED,
        )
    )

    assert "Ответ на вопрос не найден в документах." in text
    assert "ℹ️ Ответ не найден в документах." not in text
    assert "🔎 Где искали" in text
    assert "• policy.pdf — стр. 2" in text
    assert "✅ Ответ подтверждён документацией" not in text


def test_format_sources_message_shows_deduplicated_sources_and_quotes() -> None:
    text = format_sources_message(ask_response())

    assert text.count("• policy.pdf — стр. 2") == 1
    assert "• appendix.pdf — стр. 5" in text
    assert '"Отчёт нужно подать не позднее 31 июля текущего года."' in text
    assert '"Дублирующий фрагмент с той же страницы."' not in text


def test_format_sources_message_uses_search_title_for_not_found_answer() -> None:
    text = format_sources_message(ask_response(answer_status=AnswerStatus.NOT_FOUND))

    assert text.startswith("🔎 Где искали")
    assert "• policy.pdf — стр. 2" in text


def test_format_verification_details_shows_details_without_confidence() -> None:
    text = format_verification_details(
        ask_response(
            verdict=VerificationVerdict.UNSUPPORTED,
            unsupported_claims=[
                BotUnsupportedClaim(
                    claim="Нет подтверждения даты подписания.",
                    reason="В источниках нет даты подписания.",
                )
            ],
            missing_information=["Не найден договор."],
            confidence=0.35,
        )
    )

    assert "Статус: не подтверждено" in text
    assert "Неподтверждённые утверждения:" in text
    assert "• Нет подтверждения даты подписания." in text
    assert "  Причина: В источниках нет даты подписания." in text
    assert "Недостающая информация:" in text
    assert "• Не найден договор." in text
    assert "0.35" not in text
    assert "35%" not in text


def test_format_verification_details_shows_neutral_status_for_missing_verdict() -> None:
    text = format_verification_details(ask_response(verdict=None))

    assert "Статус: не проверено" in text


def test_format_verification_details_shows_not_found_status() -> None:
    text = format_verification_details(
        ask_response(
            answer_status=AnswerStatus.NOT_FOUND,
            verdict=VerificationVerdict.SUPPORTED,
        )
    )

    assert "Статус: ответ не найден в документах" in text


def test_split_telegram_text_splits_long_messages() -> None:
    parts = split_telegram_text("a" * 5000, limit=4096)

    assert len(parts) == 2
    assert "".join(parts) == "a" * 5000
    assert all(len(part) <= 4096 for part in parts)
