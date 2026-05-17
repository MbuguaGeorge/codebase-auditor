import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Text,
    Boolean,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from tracer.models import TraceEvent, ToolCall, TokenUsage, PipelineSession
from config import settings

Base = declarative_base()


class TraceEventDB(Base):
    __tablename__ = "trace_events"

    event_id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    agent_name = Column(String, nullable=False)
    iteration = Column(Integer, nullable=False, default=1)
    system_prompt = Column(Text, nullable=False)
    user_message = Column(Text, nullable=False)
    raw_response = Column(Text, nullable=True)
    stop_reason = Column(String, nullable=True)
    tool_calls_json = Column(
        Text, nullable=True, default="[]"
    )  # Store list of ToolCall as JSON
    token_usage_json = Column(
        Text, nullable=True, default="{}"
    )  # Store TokenUsage as JSON
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)


class PipelineSessionDB(Base):
    __tablename__ = "pipeline_sessions"

    session_id = Column(String, primary_key=True)
    repo_path = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    total_input_tokens = Column(Integer, nullable=False, default=0)
    total_output_tokens = Column(Integer, nullable=False, default=0)
    total_events = Column(Integer, nullable=False, default=0)
    total_duration_ms = Column(Integer, nullable=False, default=0)
    status = Column(
        String, nullable=False, default="in_progress"
    )  # in_progress, completed, failed
    error_message = Column(Text, nullable=True)


class TracerStorage:
    """
    The public interface for all database operations in the tracer.

    Usage:
        storage = TracerStorage()
        storage.save_event(trace_event)
        events = storage.get_events_by_session("some-session-id")
    """

    def __init__(self):
        self.engine = create_engine(
            settings.DATABASE_URL,
            echo=False,
        )
        Base.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)

    def _pydantic_to_db_event(self, event: TraceEvent) -> TraceEventDB:
        """
        Converts a Pydantic TraceEvent into a SQLAlchemy TraceEventDB row.
        """
        return TraceEventDB(
            event_id=event.event_id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            duration_ms=event.duration_ms,
            agent_name=event.agent_name,
            iteration=event.iteration,
            system_prompt=event.system_prompt,
            user_message=event.user_message,
            raw_response=event.raw_response,
            stop_reason=event.stop_reason,
            tool_calls_json=json.dumps([call.dict() for call in event.tool_calls]),
            token_usage_json=json.dumps(event.token_usage.model_dump()),
            success=event.success,
            error_message=event.error_message,
        )

    def _db_to_pydantic_event(self, db_event: TraceEventDB) -> TraceEvent:
        """
        Converts a SQLAlchemy TraceEventDB row back into a Pydantic TraceEvent.
        """
        return TraceEvent(
            event_id=db_event.event_id,
            session_id=db_event.session_id,
            timestamp=db_event.timestamp,
            duration_ms=db_event.duration_ms,
            agent_name=db_event.agent_name,
            iteration=db_event.iteration,
            system_prompt=db_event.system_prompt,
            user_message=db_event.user_message,
            raw_response=db_event.raw_response,
            stop_reason=db_event.stop_reason,
            tool_calls=[
                ToolCall(**call) for call in json.loads(db_event.tool_calls_json)
            ],
            token_usage=TokenUsage(**json.loads(db_event.token_usage_json)),
            success=db_event.success,
            error_message=db_event.error_message,
        )

    def _pydantic_to_db_session(self, session: PipelineSession) -> PipelineSessionDB:
        """
        Converts a Pydantic PipelineSession into a SQLAlchemy PipelineSessionDB row.
        """
        return PipelineSessionDB(
            session_id=session.session_id,
            repo_path=session.repo_path,
            started_at=session.started_at,
            completed_at=session.completed_at,
            total_input_tokens=session.total_input_tokens,
            total_output_tokens=session.total_output_tokens,
            total_events=session.total_events,
            total_duration_ms=session.total_duration_ms,
            status=session.status,
            error_message=session.error_message,
        )

    def _db_to_pydantic_session(self, db_session: PipelineSessionDB) -> PipelineSession:
        """
        Converts a SQLAlchemy PipelineSessionDB row back into a Pydantic PipelineSession.
        """
        return PipelineSession(
            session_id=db_session.session_id,
            repo_path=db_session.repo_path,
            started_at=db_session.started_at,
            completed_at=db_session.completed_at,
            total_input_tokens=db_session.total_input_tokens,
            total_output_tokens=db_session.total_output_tokens,
            total_events=db_session.total_events,
            total_duration_ms=db_session.total_duration_ms,
            status=db_session.status,
            error_message=db_session.error_message,
        )

    def save_event(self, event: TraceEvent):
        """
        Saves a TraceEvent to the database.
        """
        db_event = self._pydantic_to_db_event(event)
        with self.SessionFactory() as session:
            session.add(db_event)
            session.commit()

    def get_events_by_session(self, session_id: str) -> list[TraceEvent]:
        """
        Returns all trace events for a specific pipeline run.
        """
        with self.SessionFactory() as db:
            rows = (
                db.query(TraceEventDB)
                .filter_by(session_id=session_id)
                .order_by(TraceEventDB.timestamp.asc())
                .all()
            )
            return [self._db_to_pydantic_event(row) for row in rows]

    def get_events_by_agent(self, session_id: str, agent_name: str) -> list[TraceEvent]:
        """
        Returns all trace events for a specific agent within a pipeline run.
        """
        with self.SessionFactory() as db:
            rows = (
                db.query(TraceEventDB)
                .filter(
                    TraceEventDB.session_id == session_id,
                    TraceEventDB.agent_name == agent_name,
                )
                .order_by(TraceEventDB.timestamp.asc())
                .all()
            )
            return [self._db_to_pydantic_event(row) for row in rows]

    def get_all_events(self) -> list[TraceEvent]:
        """
        Returns all trace events across all pipeline runs.
        """
        with self.SessionFactory() as db:
            rows = db.query(TraceEventDB).order_by(TraceEventDB.timestamp.asc()).all()
            return [self._db_to_pydantic_event(row) for row in rows]

    def save_session(self, session: PipelineSession):
        """
        Saves a PipelineSession to the database.
        """
        db_session = self._pydantic_to_db_session(session)
        with self.SessionFactory() as session:
            session.add(db_session)
            session.commit()

    def update_session(self, session: PipelineSession) -> None:
        """
        Updates an existing PipelineSession in the database.
        Called at the end of orchestrator.py when the pipeline completes or fails.

        Uses merge() which handles both insert and update automatically.
        If the row exists it updates it. If not it creates it.
        """
        with self.SessionFactory() as db:
            db_session = self._pydantic_to_db_session(session)
            db.merge(db_session)
            db.commit()

    def get_session(self, session_id: str) -> PipelineSession:
        """
        Retrieves a PipelineSession for a given session_id.
        """
        with self.SessionFactory() as db:
            row = (
                db.query(PipelineSessionDB)
                .filter(PipelineSessionDB.session_id == session_id)
                .first()
            )
            if row is None:
                return None
            return self._db_to_pydantic_session(row)

    def get_all_sessions(self) -> list[PipelineSession]:
        """
        Retrieves all PipelineSessions.
        """
        with self.SessionFactory() as db:
            rows = (
                db.query(PipelineSessionDB).order_by(PipelineSessionDB.started_at).all()
            )
            return [self._db_to_pydantic_session(row) for row in rows]

    def get_token_summary(self, session_id: str) -> dict:
        """
        Returns a summary of token usage for one pipeline run.
        Broken down by agent so you can see which agent consumed the most tokens.
        """

        events = self.get_events_by_session(session_id)

        summary = {
            "session_id": session_id,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "by_agent": {},
        }

        for event in events:
            summary["total_input_tokens"] += event.token_usage.input_tokens
            summary["total_output_tokens"] += event.token_usage.output_tokens
            summary["total_tokens"] += event.token_usage.total_tokens

            agent = event.agent_name
            if agent not in summary["by_agent"]:
                summary["by_agent"][agent] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }

            summary["by_agent"][agent]["input_tokens"] += event.token_usage.input_tokens
            summary["by_agent"][agent][
                "output_tokens"
            ] += event.token_usage.output_tokens
            summary["by_agent"][agent]["total_tokens"] += event.token_usage.total_tokens
            summary["by_agent"][agent]["call_count"] += 1

        return summary
