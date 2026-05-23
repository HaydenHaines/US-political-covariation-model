"use client";

import { useCallback, useEffect, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

export interface UrlFieldSpec<T, S = unknown> {
  /** URL param name. */
  param: string;
  /** Value treated as "no param present in the URL". */
  defaultValue: T;
  /**
   * Parse the raw param value, or null when absent, into T.
   * Return defaultValue for invalid input so the hook can canonicalise it.
   * The full query is provided for fields whose fallback depends on another
   * URL value, such as PollsTable's dir default depending on sort.
   */
  parse: (raw: string | null, searchParams: URLSearchParams) => T;
  /**
   * Serialise T into URL form. Return null to omit the param.
   * The full state is provided for fields whose omission depends on another
   * field, such as preserving sort=date when dir=asc.
   */
  serialize: (value: T, state: S) => string | null;
}

export type UrlStateSpec<S> = { [K in keyof S]: UrlFieldSpec<S[K], S> };

export interface UseUrlStateResult<S> {
  state: S;
  update: (next: Partial<S>) => void;
  reset: () => void;
}

type SpecKey<S> = Extract<keyof S, string>;

function specKeys<S extends object>(spec: UrlStateSpec<S>): SpecKey<S>[] {
  return Object.keys(spec) as SpecKey<S>[];
}

function writeSpecValue<S extends object, K extends SpecKey<S>>(
  params: URLSearchParams,
  spec: UrlStateSpec<S>,
  key: K,
  state: S,
) {
  const field = spec[key];
  const serializedValue = field.serialize(state[key], state);

  if (serializedValue === null) {
    params.delete(field.param);
  } else {
    params.set(field.param, serializedValue);
  }
}

function buildUrl(pathname: string, params: URLSearchParams): string {
  const query = params.toString();
  return query ? `${pathname}?${query}` : pathname;
}

/**
 * Derive object state from configured URL params and write updates back with
 * canonical omit-at-default semantics.
 *
 * Keep the spec reference stable, usually by defining it outside the component.
 * Example:
 *
 * const { state, update } = useUrlState(POLLS_URL_SPEC);
 * update({ sort: "race", dir: "asc" });
 */
export function useUrlState<S extends object>(
  spec: UrlStateSpec<S>,
): UseUrlStateResult<S> {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const state = useMemo(() => {
    const next = {} as S;
    for (const key of specKeys(spec)) {
      next[key] = spec[key].parse(searchParams.get(spec[key].param), searchParams);
    }
    return next;
  }, [searchParams, spec]);

  const canonicalParams = useCallback(
    (nextState: S) => {
      const params = new URLSearchParams(searchParams.toString());
      for (const key of specKeys(spec)) {
        writeSpecValue(params, spec, key, nextState);
      }
      return params;
    },
    [searchParams, spec],
  );

  useEffect(() => {
    const params = canonicalParams(state);
    if (params.toString() === searchParams.toString()) return;
    router.replace(buildUrl(pathname, params), { scroll: false });
  }, [canonicalParams, pathname, router, searchParams, state]);

  const update = useCallback(
    (next: Partial<S>) => {
      const merged = { ...state, ...next };
      const params = canonicalParams(merged);
      router.replace(buildUrl(pathname, params), { scroll: false });
    },
    [canonicalParams, pathname, router, state],
  );

  const reset = useCallback(() => {
    const params = new URLSearchParams(searchParams.toString());
    for (const key of specKeys(spec)) {
      params.delete(spec[key].param);
    }
    router.replace(buildUrl(pathname, params), { scroll: false });
  }, [pathname, router, searchParams, spec]);

  return { state, update, reset };
}
