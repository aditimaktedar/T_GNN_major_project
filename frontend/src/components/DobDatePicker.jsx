import React, { useMemo, useState } from 'react';
import {
  MAX_DOB_AGE_YEARS,
  defaultViewFromValue,
  daysInMonth,
  formatIsoDate,
  parseIsoDate,
  todayIsoLocal,
} from './dobDateUtils';
import './DobDatePicker.css';

const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

const WEEKDAYS = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];

export function DobDatePicker({ id, value, onChange, invalid = false, disabled = false }) {
  const parsed = parseIsoDate(value);
  const maxIso = todayIsoLocal();
  const minYear = new Date().getFullYear() - MAX_DOB_AGE_YEARS;
  const minIso = `${minYear}-01-01`;

  const initialView = defaultViewFromValue(value);
  const [viewYear, setViewYear] = useState(initialView.year);
  const [viewMonth, setViewMonth] = useState(initialView.month);

  const years = useMemo(() => {
    const maxY = new Date().getFullYear();
    const list = [];
    for (let y = maxY; y >= minYear; y -= 1) {
      list.push(y);
    }
    return list;
  }, [minYear]);

  const calendarCells = useMemo(() => {
    const firstDow = new Date(viewYear, viewMonth, 1).getDay();
    const totalDays = daysInMonth(viewYear, viewMonth);
    const cells = [];
    for (let i = 0; i < firstDow; i += 1) {
      cells.push({ type: 'blank', key: `b-${i}` });
    }
    for (let day = 1; day <= totalDays; day += 1) {
      cells.push({ type: 'day', day, key: `d-${day}` });
    }
    return cells;
  }, [viewYear, viewMonth]);

  const commitDay = (day) => {
    const iso = formatIsoDate(viewYear, viewMonth, day);
    if (iso > maxIso || iso < minIso) return;
    onChange(iso);
  };

  const tryApplyDayInView = (year, monthIndex, day) => {
    const clampedDay = Math.min(day, daysInMonth(year, monthIndex));
    const iso = formatIsoDate(year, monthIndex, clampedDay);
    if (iso <= maxIso && iso >= minIso) {
      onChange(iso);
    }
  };

  const handleMonthSelect = (e) => {
    const month = Number(e.target.value);
    setViewMonth(month);
    if (parsed) {
      tryApplyDayInView(viewYear, month, parsed.day);
    }
  };

  const handleYearSelect = (e) => {
    const year = Number(e.target.value);
    setViewYear(year);
    if (parsed) {
      tryApplyDayInView(year, viewMonth, parsed.day);
    }
  };

  const selectedLabel = parsed
    ? `${parsed.day} ${MONTHS[parsed.month]} ${parsed.year}`
    : 'No date selected';

  return (
    <div
      className={`dob-date-picker ${invalid ? 'dob-date-picker-invalid' : ''}`}
      id={id}
      role="group"
      aria-label="Date of birth"
    >
      <div className="dob-date-picker-summary" aria-live="polite">
        {selectedLabel}
      </div>

      <div className="dob-date-picker-controls">
        <select
          className="dob-month-select"
          aria-label="Birth month"
          value={viewMonth}
          onChange={handleMonthSelect}
          disabled={disabled}
        >
          {MONTHS.map((name, index) => (
            <option key={name} value={index}>
              {name}
            </option>
          ))}
        </select>
        <select
          className="dob-year-select"
          aria-label="Birth year"
          value={viewYear}
          onChange={handleYearSelect}
          disabled={disabled}
        >
          {years.map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>
      </div>

      <div className="dob-calendar" role="application" aria-label="Birth day calendar">
        <div className="dob-calendar-weekdays">
          {WEEKDAYS.map((wd) => (
            <span key={wd} className="dob-weekday" aria-hidden="true">
              {wd}
            </span>
          ))}
        </div>
        <div className="dob-calendar-grid">
          {calendarCells.map((cell) => {
            if (cell.type === 'blank') {
              return <span key={cell.key} className="dob-day dob-day-blank" aria-hidden="true" />;
            }
            const iso = formatIsoDate(viewYear, viewMonth, cell.day);
            const isFuture = iso > maxIso;
            const isTooOld = iso < minIso;
            const isDisabled = isFuture || isTooOld || disabled;
            const isSelected =
              parsed &&
              parsed.year === viewYear &&
              parsed.month === viewMonth &&
              parsed.day === cell.day;

            return (
              <button
                key={cell.key}
                type="button"
                className={`dob-day ${isSelected ? 'dob-day-selected' : ''}`}
                disabled={isDisabled}
                aria-label={`${cell.day} ${MONTHS[viewMonth]} ${viewYear}`}
                aria-pressed={isSelected}
                onClick={() => commitDay(cell.day)}
              >
                {cell.day}
              </button>
            );
          })}
        </div>
      </div>

      {value ? (
        <button
          type="button"
          className="dob-clear-btn"
          onClick={() => onChange('')}
          disabled={disabled}
        >
          Clear date (enter age manually)
        </button>
      ) : null}
    </div>
  );
}
