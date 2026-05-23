import assert from "node:assert/strict";
import type { UrlStateSpec } from "./use-url-state";

type Sort = "date" | "race";
type Dir = "asc" | "desc";

interface TestState {
  sort: Sort;
  dir: Dir;
}

const defaultSort: Sort = "date";
const defaultDir: Dir = "desc";

function isSort(value: string | null): value is Sort {
  return value === "date" || value === "race";
}

function isDir(value: string | null): value is Dir {
  return value === "asc" || value === "desc";
}

function defaultDirForSort(sort: Sort): Dir {
  return sort === "date" ? "desc" : "asc";
}

const spec: UrlStateSpec<TestState> = {
  sort: {
    param: "sort",
    defaultValue: defaultSort,
    parse: (raw) => (isSort(raw) ? raw : defaultSort),
    serialize: (value, state) => (value === defaultSort && state.dir === defaultDir ? null : value),
  },
  dir: {
    param: "dir",
    defaultValue: defaultDir,
    parse: (raw, params) => {
      const sort = params.get("sort");
      const sortValue = isSort(sort) ? sort : defaultSort;
      return isSort(sort) && isDir(raw) ? raw : defaultDirForSort(sortValue);
    },
    serialize: (value, state) => (state.sort === defaultSort && value === defaultDir ? null : value),
  },
};

function parse(query: string): TestState {
  const params = new URLSearchParams(query);
  return {
    sort: spec.sort.parse(params.get("sort"), params),
    dir: spec.dir.parse(params.get("dir"), params),
  };
}

function canonicalize(query: string, state = parse(query)): string {
  const params = new URLSearchParams(query);
  const serializedSort = spec.sort.serialize(state.sort, state);
  if (serializedSort === null) {
    params.delete(spec.sort.param);
  } else {
    params.set(spec.sort.param, serializedSort);
  }

  const serializedDir = spec.dir.serialize(state.dir, state);
  if (serializedDir === null) {
    params.delete(spec.dir.param);
  } else {
    params.set(spec.dir.param, serializedDir);
  }
  return params.toString();
}

function update(query: string, next: Partial<TestState>): string {
  return canonicalize(query, { ...parse(query), ...next });
}

export function runUseUrlStateContractAssertions() {
  assert.deepEqual(parse("sort=race&dir=asc"), { sort: "race", dir: "asc" });
  assert.deepEqual(parse("sort=garbage&dir=asc"), { sort: "date", dir: "desc" });
  assert.equal(canonicalize("sort=date&dir=desc"), "");
  assert.equal(canonicalize("sort=garbage&dir=asc"), "");
  assert.equal(update("", { sort: "race", dir: "asc" }), "sort=race&dir=asc");
  assert.equal(update("page=2", { sort: "race", dir: "asc" }), "page=2&sort=race&dir=asc");
  assert.equal(canonicalize("sort=race"), "sort=race&dir=asc");
  assert.equal(canonicalize("sort=race&dir=asc"), "sort=race&dir=asc");
  assert.equal(canonicalize("sort=race&dir=desc"), "sort=race&dir=desc");
  assert.equal(canonicalize("sort=date&dir=asc"), "sort=date&dir=asc");
  assert.equal(canonicalize("sort=race&dir=asc", { sort: "date", dir: "desc" }), "");
}
