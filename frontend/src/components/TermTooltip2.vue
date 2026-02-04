<!-- src/components/TermTooltip.vue -->
<!-- VERSION 3.0 - APPEARS ABOVE SCROLLBARS - Extreme z-index for ALL elements -->
<template>
  <span class="term-tooltip-wrapper" ref="wrapperRef">
    <span 
      class="term-highlight"
      @mouseenter="handleMouseEnter"
      @mouseleave="handleMouseLeave"
      @click.prevent="toggleTooltip"
    >
      {{ term }}
    </span>
    <Teleport to="body">
      <Transition name="tooltip-fade">
        <div 
          v-if="showTooltip" 
          class="tooltip-content"
          :style="tooltipStyle"
          ref="tooltipRef"
          @mouseenter="showTooltip = true"
          @mouseleave="showTooltip = false"
        >
          <div class="tooltip-header">{{ term }} - {{ termDefinitions[term]?.title }}</div>
          <div class="tooltip-body">
            <div v-html="formattedDefinition"></div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </span>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue';

const props = defineProps({
  term: {
    type: String,
    required: true
  }
});

const showTooltip = ref(false);
const wrapperRef = ref(null);
const tooltipRef = ref(null);
const tooltipStyle = ref({});

const handleMouseEnter = async () => {
  showTooltip.value = true;
  await nextTick();
  positionTooltip();
};

const handleMouseLeave = () => {
  showTooltip.value = false;
};

const toggleTooltip = async () => {
  showTooltip.value = !showTooltip.value;
  if (showTooltip.value) {
    await nextTick();
    positionTooltip();
  }
};

const positionTooltip = () => {
  if (!wrapperRef.value || !tooltipRef.value) return;
  
  const triggerRect = wrapperRef.value.getBoundingClientRect();
  const tooltipRect = tooltipRef.value.getBoundingClientRect();
  
  // ALWAYS position tooltip BELOW the trigger element (roll down)
  const top = triggerRect.bottom + 8;
  
  // Center the tooltip horizontally relative to the trigger
  let left = triggerRect.left + (triggerRect.width / 2) - (tooltipRect.width / 2);
  
  // Ensure tooltip doesn't go off the left edge of the screen
  if (left < 10) {
    left = 10;
  }
  
  // Ensure tooltip doesn't go off the right edge of the screen
  if (left + tooltipRect.width > window.innerWidth - 10) {
    left = window.innerWidth - tooltipRect.width - 10;
  }
  
  // Only position above if tooltip would go significantly off bottom AND header is in lower half of screen
  const wouldGoOffBottom = top + tooltipRect.height > window.innerHeight - 10;
  const isInLowerHalf = triggerRect.top > (window.innerHeight / 2);
  
  if (wouldGoOffBottom && isInLowerHalf) {
    // Position above only if in lower half of screen
    const topAlt = triggerRect.top - tooltipRect.height - 8;
    tooltipStyle.value = {
      top: `${topAlt}px`,
      left: `${left}px`
    };
  } else {
    // Default: ALWAYS roll down
    tooltipStyle.value = {
      top: `${top}px`,
      left: `${left}px`
    };
  }
};

const termDefinitions = {
  'ORC': {
    title: 'Offer Release Code',
    definition: `Indicates when a shipment will be released:
    
<strong>Automatic release</strong> of shipment by the shipping activity, without advance notice of availability (NOA).

<strong>X</strong> - If corresponding X in MILSTRIP, material will be moved via the Defense Transportation System. If corresponding W in MILSTRIP, shipping office must contact the program office for correct location.

<strong>Y</strong> - NOA required but shipment may be released automatically if no response is received within 15 calendar days.

<strong>Z</strong> - NOA is required and shipment cannot be released until a response is received.`
  },
  
  'DTC': {
    title: 'Delivery Term Code',
    definition: `Specifies how and where defense articles are delivered:

<strong>2</strong> - Inland to Inland movement

<strong>4</strong> - Foreign partner is responsible for CONUS transportation and onward movement

<strong>5</strong> - U.S. movement for FMS customer within CONUS/Canada

<strong>7</strong> - DoD movement from point of origin to, and including, inland carrier delivery to the specified inland location — for overseas shipments:
<div style="margin-left: 20px;">
  a) To Europe, Hawaii, Latin America (Central America & Caribbean Basin), Mediterranean ports<br>
  b) To Newfoundland, Labrador, Thule, Iceland, South America (East & West Coasts), Far East, African ports (other than Mediterranean), Near East
</div>

<strong>8</strong> - DoD movement from point of origin to, and including, unloading, handling, and storage aboard vessel at the port of exit.

<strong>9</strong> - DoD movement from point of origin to, and including, vessel discharge at the point of discharge — i.e., delivery to overseas port of destination:
<div style="margin-left: 20px;">
  1. To Europe, Hawaii, Latin America & Mediterranean ports<br>
  2. To Newfoundland, Labrador, Thule, Iceland, South America East/West, Far East, African ports, Near East.
</div>`
  },
  
  'TA': {
    title: 'Type of Assistance',
    definition: `Codes used to indicate funding source, terms of sale, and source code:

<strong>3</strong> - Term of Sale: Cash with Acceptance or Prior to Delivery; Source Code S, R, E, or F (stock)

<strong>4</strong> - Term of Sale: Cash with Acceptance or Prior to Delivery; Source Code X (undetermined)

<strong>5</strong> - Term of Sale: Cash with Acceptance; Source Code P (procurement)

<strong>6</strong> - Term of Sale: Payment on Delivery; Source Code S, R, E, or F

<strong>7</strong> - Term of Sale: Dependable Undertaking with 120-day Payment after Delivery; Source Code P

<strong>8</strong> - Term of Sale: Payment 120 days after Delivery; Source Code S, R, E, or F

<strong>I</strong> - Term of Sale: EDA grant (Excess Defense Articles), non-reimbursable, Source Code E

<strong>U</strong> - Cooperative Logistics Supply Support Arrangement (CLSSA) Foreign Military Sales Order (FMSO) I case, Source Code P

<strong>V</strong> - CLSSA FMSO II stocks acquired under FMSO I case, Source Code S

<strong>M</strong> - Term of Sale: MAP Merger / USG Grant. Use instead of TA Codes 3-8

<strong>N</strong> - Term of Sale: FMS Credit (Non-repayable). Use instead of TA Codes 3-8

<strong>Y</strong> - Term of Sale FMF Guarantee. Use instead of TA Codes 3-8

<strong>Z</strong> - Term of Sale FMS Credit. Use instead of TA Codes 3-8`
  },
  
  'SC': {
    title: 'Source of Supply Code',
    definition: `How the item/service is being sourced:

<strong>S</strong> - Shipment from DoD stocks or performance by DoD personnel

<strong>P</strong> - From new procurement

<strong>R</strong> - From rebuild, repair, or modification by the USG

<strong>X</strong> - Mixed source, such as stock and procurement, or undetermined

<strong>E</strong> - Excess items, as is

<strong>F</strong> - Special Defense Acquisition Fund (SDAF) items`
  }
};

const formattedDefinition = computed(() => {
  const def = termDefinitions[props.term]?.definition || '';
  return def.trim();
});
</script>

<style scoped>
.term-tooltip-wrapper {
  position: relative;
  display: inline-block;
}

.term-highlight {
  color: white;
  font-weight: 600;
  cursor: help;
  text-decoration: underline;
  text-decoration-style: dotted;
  text-decoration-color: rgba(255, 255, 255, 0.6);
  padding: 2px 4px;
  border-radius: 3px;
  transition: all 0.2s ease;
}

.term-highlight:hover {
  background-color: rgba(255, 255, 255, 0.1);
  text-decoration-color: white;
}

/* CRITICAL: EXTREME z-index to appear above scrollbars and ALL UI elements */
/* Using Teleport to body ensures no parent clipping */
.tooltip-content {
  position: fixed !important;
  z-index: 2147483647 !important; /* Maximum 32-bit integer - highest possible z-index */
  background: white !important;
  border: 2px solid #2c5f7f !important;
  border-radius: 8px !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
  padding: 0 !important;
  width: 380px !important;
  max-width: 90vw !important;
  pointer-events: auto !important;
  isolation: isolate !important;
  will-change: transform !important;
}

.tooltip-header {
  font-weight: 700;
  font-size: 15px;
  color: white;
  background: #2c5f7f;
  padding: 12px 16px;
  margin: 0;
  border-radius: 6px 6px 0 0;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
  line-height: 1.4;
}

.tooltip-body {
  font-size: 13px;
  color: #374151;
  line-height: 1.7;
  max-height: 500px;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 16px;
  word-wrap: break-word;
  overflow-wrap: break-word;
}

.tooltip-body :deep(strong) {
  color: #1f2937;
  font-weight: 700;
}

.tooltip-body :deep(div) {
  margin: 8px 0;
  word-wrap: break-word;
  overflow-wrap: break-word;
}

/* Smooth tooltip transition - rolls DOWN */
.tooltip-fade-enter-active,
.tooltip-fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.tooltip-fade-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.tooltip-fade-leave-to {
  opacity: 0;
  transform: translateY(5px);
}

/* Scrollbar styling */
.tooltip-body::-webkit-scrollbar {
  width: 8px;
}

.tooltip-body::-webkit-scrollbar-track {
  background: #f3f4f6;
  border-radius: 4px;
}

.tooltip-body::-webkit-scrollbar-thumb {
  background: #9ca3af;
  border-radius: 4px;
}

.tooltip-body::-webkit-scrollbar-thumb:hover {
  background: #6b7280;
}
</style>