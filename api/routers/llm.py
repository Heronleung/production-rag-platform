"""Runtime LLM provider readiness checks for the settings interface."""

from fastapi import APIRouter, HTTPException, status

from api.llm import get_llm
from api.schemas import LLMCheckRequest, LLMCheckResponse

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/check", response_model=LLMCheckResponse)
def check_llm(payload: LLMCheckRequest) -> LLMCheckResponse:
    try:
        chat_model = get_llm(provider=payload.provider, model=payload.model)
        detail = chat_model.check_ready()
    except (ImportError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    return LLMCheckResponse(
        ok=True,
        provider=payload.provider,
        model=chat_model.model,
        detail=detail,
    )
