"use client";

import { useState, useRef, useEffect } from "react";

interface RaceOption {
  slug: string;
  label: string;
}

interface RaceSelectorProps {
  id?: string;
  options: RaceOption[];
  value: string;
  placeholder?: string;
  disabled?: boolean;
  onChange: (slug: string) => void;
}

/**
 * Searchable combobox for race selection.
 *
 * Renders a text input that filters the race list as the user types.
 * Clicking an option selects it and closes the dropdown.
 * The input shows the selected label when closed and a filter query
 * when open, so users can type to search without losing context.
 *
 * Keyboard support: Arrow keys navigate, Enter selects, Escape closes.
 */
export function RaceSelector({
  id,
  options,
  value,
  placeholder = "Select a race",
  disabled = false,
  onChange,
}: RaceSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectedOption = options.find((o) => o.slug === value);

  // Filter options by query (case-insensitive, matches slug or label)
  const filtered = query
    ? options.filter(
        (o) =>
          o.label.toLowerCase().includes(query.toLowerCase()) ||
          o.slug.toLowerCase().includes(query.toLowerCase()),
      )
    : options;

  // Close dropdown when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
        setQuery("");
        setActiveIndex(-1);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleInputFocus = () => {
    if (!disabled) {
      setIsOpen(true);
      setQuery("");
      setActiveIndex(-1);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    setIsOpen(true);
    setActiveIndex(-1);
  };

  const handleSelect = (option: RaceOption) => {
    onChange(option.slug);
    setIsOpen(false);
    setQuery("");
    setActiveIndex(-1);
    inputRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen) {
      if (e.key === "ArrowDown" || e.key === "Enter") {
        setIsOpen(true);
      }
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (activeIndex >= 0 && filtered[activeIndex]) {
        handleSelect(filtered[activeIndex]);
      }
    } else if (e.key === "Escape") {
      setIsOpen(false);
      setQuery("");
      setActiveIndex(-1);
    }
  };

  const displayValue = isOpen ? query : (selectedOption?.label ?? "");

  return (
    <div ref={containerRef} className="relative">
      <input
        ref={inputRef}
        id={id}
        type="text"
        role="combobox"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-autocomplete="list"
        autoComplete="off"
        value={displayValue}
        placeholder={placeholder}
        disabled={disabled}
        onChange={handleInputChange}
        onFocus={handleInputFocus}
        onKeyDown={handleKeyDown}
        className="w-full rounded border px-3 py-2 text-sm focus:outline-none focus:ring-2"
        style={{
          background: disabled ? "var(--color-border-subtle)" : "var(--color-surface)",
          border: "1px solid var(--color-border)",
          color: "var(--color-text)",
          cursor: disabled ? "not-allowed" : "text",
          // Focused ring in dem-primary color for Dusty Ink consistency
          ["--tw-ring-color" as string]: "var(--forecast-safe-d)",
        }}
      />

      {isOpen && !disabled && (
        <ul
          role="listbox"
          className="absolute z-50 mt-1 w-full max-h-60 overflow-y-auto rounded border shadow-md"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
          }}
        >
          {filtered.length === 0 ? (
            <li
              className="px-3 py-2 text-sm"
              style={{ color: "var(--color-text-muted)" }}
            >
              No races match &ldquo;{query}&rdquo;
            </li>
          ) : (
            filtered.map((option, i) => {
              const isSelected = option.slug === value;
              const isActive = i === activeIndex;
              return (
                <li
                  key={option.slug}
                  role="option"
                  aria-selected={isSelected}
                  className="px-3 py-2 text-sm cursor-pointer"
                  style={{
                    background: isActive
                      ? "var(--color-border)"
                      : isSelected
                      ? "var(--color-border-subtle)"
                      : "transparent",
                    color: isSelected
                      ? "var(--forecast-safe-d)"
                      : "var(--color-text)",
                    fontWeight: isSelected ? 600 : 400,
                  }}
                  onMouseDown={(e) => {
                    // Prevent input blur before click registers
                    e.preventDefault();
                    handleSelect(option);
                  }}
                  onMouseEnter={() => setActiveIndex(i)}
                >
                  {option.label}
                </li>
              );
            })
          )}
        </ul>
      )}
    </div>
  );
}
