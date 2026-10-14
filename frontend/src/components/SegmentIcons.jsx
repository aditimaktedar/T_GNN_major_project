import React from 'react';

const iconProps = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  'aria-hidden': true,
};

export function MedicationsIcon({ className = 'segment-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <rect x="3" y="8" width="18" height="8" rx="4" />
      <path d="M7 8V6a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2" strokeLinecap="round" />
    </svg>
  );
}

export function RagIcon({ className = 'segment-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <path d="M9 3h6v4H9zM5 7h14v14H5z" strokeLinejoin="round" />
      <path d="M9 12h6M9 16h4" strokeLinecap="round" />
    </svg>
  );
}

export function AllSectionsIcon({ className = 'segment-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <path d="M4 6h16M4 12h16M4 18h10" strokeLinecap="round" />
    </svg>
  );
}

export function PillLabelIcon({ className = 'lbl-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <rect x="3" y="8" width="18" height="8" rx="4" />
    </svg>
  );
}

export function BoltLabelIcon({ className = 'lbl-icon-svg' }) {
  return (
    <svg className={className} {...iconProps} fill="currentColor" stroke="none">
      <path d="M13 2L3 14h7v8l10-12h-7z" />
    </svg>
  );
}

export function FolderIcon({ className = 'branch-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <path d="M3 7h6l2 2h10v10H3z" strokeLinejoin="round" />
    </svg>
  );
}

export function BookIcon({ className = 'branch-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <path d="M5 4h12a1 1 0 0 1 1 1v15l-7-3-7 3V5a1 1 0 0 1 1-1z" />
    </svg>
  );
}

export function DnaIcon({ className = 'branch-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <path d="M7 4c3 4 3 12 0 16M17 4c-3 4-3 12 0 16M8 8h8M8 16h8" strokeLinecap="round" />
    </svg>
  );
}

export function ChartIcon({ className = 'branch-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <path d="M4 19V5M4 19h16M8 16V10M12 16V7M16 16v-4" strokeLinecap="round" />
    </svg>
  );
}

export function UserNavIcon({ className = 'nav-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <circle cx="12" cy="8" r="3" />
      <path d="M5 20c1.5-3 4-4.5 7-4.5s5.5 1.5 7 4.5" strokeLinecap="round" />
    </svg>
  );
}

export function ShieldIcon({ className = 'branch-icon-svg' }) {
  return (
    <svg className={className} {...iconProps}>
      <path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" />
    </svg>
  );
}
