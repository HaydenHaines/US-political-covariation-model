import assert from "node:assert/strict";
import type { UrlStateSpec } from "./use-url-state";

type Sort = "date" | "race";
type Dir = "asc" | "desc";

interface TestState {
  sort: Sort;
  dir: Dir;
}

interface PageState {
  page: number;
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

const pageSpec: UrlStateSpec<PageState> = {
  page: {
    param: "page",
    defaultValue: 0,
    parse: (raw) => {
      const page = Number(raw);
      return Number.isInteger(page) && page >= 0 ? page : 0;
    },
    serialize: (value) => (value === 0 ? null : String(value)),
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

function parsePage(query: string): PageState {
  const params = new URLSearchParams(query);
  return {
    page: pageSpec.page.parse(params.get("page"), params),
  };
}

function canonicalizePage(query: string): string {
  const params = new URLSearchParams(query);
  const state = parsePage(query);
  const serializedPage = pageSpec.page.serialize(state.page, state);
  if (serializedPage === null) {
    params.delete(pageSpec.page.param);
  } else {
    params.set(pageSpec.page.param, serializedPage);
  }
  return params.toString();
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
  assert.equal(canonicalizePage("page=0"), "");
  assert.equal(canonicalizePage("page=2"), "page=2");
  assert.equal(canonicalizePage("page=bogus"), "");
}
