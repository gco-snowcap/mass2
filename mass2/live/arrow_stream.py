"""Thin wrappers that let polars DataFrames be written to, and tailed from, a growing Arrow IPC *stream* file.

Why the stream format and not the Arrow IPC *file* format (Feather v2)? The file format puts its schema and
record-batch index in a footer that is written only on close, so a reader cannot use a file that is still
being written. The stream format is just a sequence of self-delimiting messages (schema, batch, batch, ...,
end-of-stream marker). Each message is appended as soon as it is ready, so a reader can pick up every complete
batch while the writer is still running. By convention such files use the extension ".arrows".

polars can write and read a *complete* stream (`DataFrame.write_ipc_stream`, `pl.read_ipc_stream`), but it can
neither append batches to an open stream nor read one incrementally. These classes add exactly those two
operations using pyarrow, and otherwise traffic only in polars DataFrames.
"""

import os
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import polars as pl
import pyarrow as pa

# Every IPC message starts with this 4-byte continuation marker followed by an int32 metadata length.
# A metadata length of zero is the end-of-stream marker.
_CONTINUATION = b"\xff\xff\xff\xff"
_END_OF_STREAM = _CONTINUATION + b"\x00\x00\x00\x00"


class ArrowStreamWriter:
    """Append polars DataFrames, one record batch each, to an Arrow IPC stream file.

    The first DataFrame fixes the schema. Later frames are conformed to it: columns are reordered and cast,
    missing columns are filled with nulls, and unexpected columns are dropped. Every batch is flushed to the
    OS immediately, so a concurrent `ArrowStreamTailer` sees it on its next poll.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "wb")
        self._writer: pa.ipc.RecordBatchStreamWriter | None = None
        self.schema: pl.Schema | None = None
        self.batches_written = 0

    def write(self, df: pl.DataFrame) -> None:
        """Append `df` as exactly one record batch."""
        if self.schema is None:
            self.schema = df.schema
        else:
            df = conform_to_schema(df, self.schema)
        table = df.to_arrow().combine_chunks()
        if self._writer is None:
            self._writer = pa.ipc.new_stream(self._file, table.schema)
        batch = table.to_batches()[0] if table.num_rows else pa.RecordBatch.from_pylist([], schema=table.schema)
        self._writer.write_batch(batch)
        self._file.flush()
        self.batches_written += 1

    @property
    def bytes_written(self) -> int:
        """Bytes in the file so far."""
        return self._file.tell() if not self._file.closed else self.path.stat().st_size

    def close(self) -> None:
        """Write the end-of-stream marker and close the file. Safe to call twice."""
        if self._file.closed:
            return
        if self._writer is not None:
            self._writer.close()  # writes the end-of-stream marker
        else:
            self._file.write(_END_OF_STREAM)  # an empty, but valid and finished, stream
        self._file.close()

    def __enter__(self) -> "ArrowStreamWriter":
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()


class ArrowStreamTailer:
    """Read the complete record batches that have been appended to an Arrow IPC stream file since the last poll.

    The file need not exist yet, and its last message may be only partly written: incomplete bytes are kept
    and retried on the next poll. `ended` becomes True once the end-of-stream marker has been read. Each poll
    reads at most `max_read_bytes` of new data, so starting far behind (or on a finished file) uses bounded memory.
    """

    def __init__(self, path: str | Path, max_read_bytes: int = 64 * 1024 * 1024):
        self.path = Path(path)
        self.max_read_bytes = max_read_bytes
        self._file: BinaryIO | None = None
        self._pending = b""  # bytes read from the file but not yet parsed into complete messages
        self._schema: pa.Schema | None = None
        self.ended = False
        self.batches_read = 0
        self.bytes_read = 0  # bytes of complete messages consumed

    def poll(self) -> list[pl.DataFrame]:
        """Return a list of DataFrames, one per new complete record batch (possibly an empty list)."""
        if self.ended:
            return []
        if self._file is None:
            if not self.path.exists():
                return []
            self._file = open(self.path, "rb")
        self._pending += self._file.read(self.max_read_bytes)

        frames = []
        buf = pa.py_buffer(self._pending)
        pos = 0
        while True:
            header = self._pending[pos : pos + 8]
            if len(header) < 8:
                break
            if header == _END_OF_STREAM:
                self.ended = True
                pos += 8
                break
            reader = pa.BufferReader(buf[pos:])
            try:
                message = pa.ipc.read_message(reader)
            except (pa.ArrowInvalid, OSError, EOFError):
                break  # the writer has not finished this message yet
            pos += reader.tell()
            if message.type == "schema":
                self._schema = pa.ipc.read_schema(message)
            elif message.type == "record batch":
                batch = pa.ipc.read_record_batch(message, self._schema)
                frames.append(pl.DataFrame(pl.from_arrow(batch)))
                self.batches_read += 1
            else:
                raise ValueError(f"Unsupported Arrow IPC message type {message.type!r} in {self.path}")
        self._pending = self._pending[pos:]
        self.bytes_read += pos
        if self.ended:
            self.close()
        return frames

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None


def conform_to_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Return `df` with exactly the columns of `schema`, in order, cast to its dtypes; missing columns are null."""
    return df.select([
        pl.col(name).cast(dtype) if name in df.columns else pl.lit(None, dtype).alias(name) for name, dtype in schema.items()
    ])


def write_stream_atomically(df: pl.DataFrame, path: str | Path) -> None:
    """Write `df` as a complete IPC stream, replacing `path` atomically so readers never see a partial file."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.write_ipc_stream(tmp)
    os.replace(tmp, path)


def write_text_atomically(text: str, path: str | Path) -> None:
    """Write `text` to `path`, replacing it atomically."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
