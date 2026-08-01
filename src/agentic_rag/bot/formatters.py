from collections.abc import Iterable

from agentic_rag.bot.client import (
    BotAskResponse,
    BotDocumentChunkResponse,
    BotUnsupportedClaim,
    BotVerificationResult,
)
from agentic_rag.core.enums import AnswerStatus, VerificationVerdict

TELEGRAM_MESSAGE_LIMIT = 4096
SOURCE_QUOTE_LIMIT = 280
TRUNCATED_TEXT_SUFFIX = "\n\n…\nСообщение слишком длинное, показана первая часть."


def format_answer_message(result: BotAskResponse) -> str:
    lines = [
        "💬 Ответ",
        "",
        result.answer.strip() or "Ответ не найден.",
        "",
        "🔎 Где искали",
        "",
    ]

    source_lines = _source_reference_lines(result.chunks)
    lines.extend(source_lines if source_lines else ["Фрагменты не найдены."])

    if result.answer_status != AnswerStatus.NOT_FOUND:
        lines.extend(["", *_format_short_verification_status(result.verification_result)])

    return "\n".join(lines)


def format_sources_message(result: BotAskResponse) -> str:
    lines = ["🔎 Где искали"]

    if not result.chunks:
        lines.extend(["", "Фрагменты не найдены."])
        return "\n".join(lines)

    for chunk in _unique_source_chunks(result.chunks):
        lines.extend(["", f"• {chunk.source} — стр. {chunk.page}"])

        quote = _short_quote(chunk.text)
        if quote:
            lines.append(f'  "{quote}"')

    return "\n".join(lines)


def format_verification_details(result: BotAskResponse) -> str:
    verification = result.verification_result
    lines = [
        "✅ Проверка",
        "",
        f"Статус: {_verification_status_text(result.answer_status, verification)}",
    ]

    if verification.unsupported_claims:
        lines.extend(["", "Неподтверждённые утверждения:"])
        lines.extend(_unsupported_claim_detail_lines(verification.unsupported_claims))

    if verification.missing_information:
        lines.extend(["", "Недостающая информация:"])
        lines.extend(_bullet_lines(verification.missing_information))

    return "\n".join(lines)


def split_telegram_text(
    text: str,
    *,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""

    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                parts.append(current.rstrip())
                current = ""

            parts.extend(line[index : index + limit] for index in range(0, len(line), limit))
            continue

        if len(current) + len(line) > limit:
            parts.append(current.rstrip())
            current = line
            continue

        current += line

    if current:
        parts.append(current.rstrip())

    return parts


def fit_telegram_text(
    text: str,
    *,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> str:
    if len(text) <= limit:
        return text

    text_limit = limit - len(TRUNCATED_TEXT_SUFFIX)
    return text[:text_limit].rstrip() + TRUNCATED_TEXT_SUFFIX


def _format_short_verification_status(
    verification: BotVerificationResult,
) -> list[str]:
    if verification.verdict is None:
        return ["ℹ️ Ответ не был проверен."]

    if verification.verdict == VerificationVerdict.SUPPORTED:
        return ["✅ Ответ подтверждён документацией"]

    lines = ["⚠️ Не все утверждения удалось подтвердить по документам."]
    if verification.unsupported_claims:
        lines.extend(["", "Не подтверждено:"])
        lines.extend(
            f"• {unsupported_claim.claim}" for unsupported_claim in verification.unsupported_claims
        )

    return lines


def _verification_status_text(
    answer_status: AnswerStatus,
    verification: BotVerificationResult,
) -> str:
    if answer_status == AnswerStatus.NOT_FOUND:
        return "ответ не найден в документах"

    if verification.verdict is None:
        return "не проверено"

    if verification.verdict == VerificationVerdict.SUPPORTED:
        return "подтверждено"

    return "не подтверждено"


def _source_reference_lines(chunks: list[BotDocumentChunkResponse]) -> list[str]:
    return [f"• {chunk.source} — стр. {chunk.page}" for chunk in _unique_source_chunks(chunks)]


def _unique_source_chunks(
    chunks: list[BotDocumentChunkResponse],
) -> list[BotDocumentChunkResponse]:
    seen: set[tuple[str, int]] = set()
    result: list[BotDocumentChunkResponse] = []

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue

        seen.add(key)
        result.append(chunk)

    return result


def _short_quote(text: str) -> str:
    quote = " ".join(text.split())
    if len(quote) <= SOURCE_QUOTE_LIMIT:
        return quote

    return quote[: SOURCE_QUOTE_LIMIT - 1].rstrip() + "…"


def _bullet_lines(items: Iterable[str]) -> list[str]:
    return [f"• {item}" for item in items if item]


def _unsupported_claim_detail_lines(
    unsupported_claims: Iterable[BotUnsupportedClaim],
) -> list[str]:
    lines: list[str] = []

    for unsupported_claim in unsupported_claims:
        claim = unsupported_claim.claim.strip()
        reason = unsupported_claim.reason.strip()
        if not claim:
            continue

        lines.append(f"• {claim}")
        if reason:
            lines.append(f"  Причина: {reason}")

    return lines
