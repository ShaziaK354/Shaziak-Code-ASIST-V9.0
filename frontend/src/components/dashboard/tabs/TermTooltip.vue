<!-- src/components/TermTooltip.vue -->
<template>
  <div class="term-tooltip-wrapper">
    <span 
      class="term-highlight"
      @mouseenter="showTooltip = true"
      @mouseleave="showTooltip = false"
    >
      {{ term }}
    </span>
    <div 
      v-if="showTooltip" 
      class="tooltip-content"
      :style="tooltipPosition"
    >
      <div class="tooltip-header">{{ term }} - {{ termDefinitions[term]?.title }}</div>
      <div class="tooltip-body">
        <div v-html="formattedDefinition"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';

const props = defineProps({
  term: {
    type: String,
    required: true
  }
});

const showTooltip = ref(false);

const termDefinitions = {
  'ORC': {
    title: 'Offer Release Code',
    definition: `Indicates when a shipment will be released:
    
- <strong>Automatic release</strong> of shipment by the shipping activity, without advance notice of availability (NOA).

- <strong>X</strong> - If corresponding X in MILSTRIP, material will be moved via the Defense Transportation System. If corresponding W in MILSTRIP, shipping office must contact the program office for correct location.

- <strong>Y</strong> - NOA required but shipment may be released automatically if no response is received within 15 calendar days.

- <strong>Z</strong> - NOA is required and shipment cannot be released until a response is received.`
  },
  
  'DTC': {
    title: 'Delivery Term Code',
    definition: `Specifies how and where defense articles are delivered:

- <strong>2</strong> - Inland to Inland movement

- <strong>4</strong> - Foreign partner is responsible for CONUS transportation and onward movement

- <strong>5</strong> - U.S. movement for FMS customer within CONUS/Canada

- <strong>7</strong> - DoD movement from point of origin to, and including, inland carrier delivery to the specified inland location — for overseas shipments:
  a) To Europe, Hawaii, Latin America (Central America & Caribbean Basin), Mediterranean ports
  b) To Newfoundland, Labrador, Thule, Iceland, South America (East & West Coasts), Far East, African ports (other than Mediterranean), Near East

- <strong>8</strong> - DoD movement from point of origin to, and including, unloading, handling, and storage aboard vessel at the port of exit.

- <strong>9</strong> - DoD movement from point of origin to, and including, vessel discharge at the point of discharge — i.e., delivery to overseas port of destination:
  1. To Europe, Hawaii, Latin America & Mediterranean ports
  2. To Newfoundland, Labrador, Thule, Iceland, South America East/West, Far East, African ports, Near East.`
  },
  
  'TA': {
    title: 'Type of Assistance',
    definition: `Codes used to indicate funding source, terms of sale, and source code:

- <strong>3</strong> - Term of Sale: Cash with Acceptance or Prior to Delivery; Source Code S, R, E, or F (stock)

- <strong>4</strong> - Term of Sale: Cash with Acceptance or Prior to Delivery; Source Code X (undetermined)

- <strong>5</strong> - Term of Sale: Cash with Acceptance; Source Code P (procurement)

- <strong>6</strong> - Term of Sale: Payment on Delivery; Source Code S, R, E, or F

- <strong>7</strong> - Term of Sale: Dependable Undertaking with 120-day Payment after Delivery; Source Code P

- <strong>8</strong> - Term of Sale: Payment 120 days after Delivery; Source Code S, R, E, or F

- <strong>I</strong> - Term of Sale: EDA grant (Excess Defense Articles), non-reimbursable, Source Code E

- <strong>U</strong> - Cooperative Logistics Supply Support Arrangement (CLSSA) Foreign Military Sales Order (FMSO) I case, Source Code P

- <strong>V</strong> - CLSSA FMSO II stocks acquired under FMSO I case, Source Code S

- <strong>M</strong> - Term of Sale: MAP Merger / USG Grant. Use instead of TA Codes 3-8

- <strong>N</strong> - Term of Sale: FMS Credit (Non-repayable). Use instead of TA Codes 3-8

- <strong>Y</strong> - Term of Sale FMF Guarantee. Use instead of TA Codes 3-8

- <strong>Z</strong> - Term of Sale FMS Credit. Use instead of TA Codes 3-8`
  },
  
  'SC': {
    title: 'Source of Supply Code',
    definition: `How the item/service is being sourced:

- <strong>S</strong> - Shipment from DoD stocks or performance by DoD personnel

- <strong>P</strong> - From new procurement

- <strong>R</strong> - From rebuild, repair, or modification by the USG

- <strong>X</strong> - Mixed source, such as stock and procurement, or undetermined

- <strong>E</strong> - Excess items, as is

- <strong>F</strong> - Special Defense Acquisition Fund (SDAF) items`
  }
};

const formattedDefinition = computed(() => {
  const def = termDefinitions[props.term]?.definition || '';
  return def.replace(/\n/g, '<br>');
});

const tooltipPosition = computed(() => {
  return {
    top: '100%',
    left: '50%',
    transform: 'translateX(-50%)'
  };
});
</script>

<style scoped>
.term-tooltip-wrapper {
  position: relative;
  display: inline-block;
}

.term-highlight {
  color: var(--color-primary);
  font-weight: 600;
  cursor: help;
  border-bottom: 1px dashed var(--color-primary);
  padding: 0 2px;
}

.term-highlight:hover {
  color: var(--color-primary-dark);
  border-bottom-color: var(--color-primary-dark);
}

.tooltip-content {
  position: absolute;
  z-index: 1000;
  background: white;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  padding: var(--space-sm);
  min-width: 400px;
  max-width: 600px;
  margin-top: var(--space-xs);
  pointer-events: none;
}

.tooltip-header {
  font-weight: 700;
  font-size: var(--font-size-base);
  color: var(--color-text-primary);
  margin-bottom: var(--space-xs);
  padding-bottom: var(--space-xs);
  border-bottom: 2px solid var(--color-primary);
}

.tooltip-body {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  line-height: 1.6;
  max-height: 400px;
  overflow-y: auto;
}

.tooltip-body :deep(strong) {
  color: var(--color-text-primary);
  font-weight: 600;
}
</style>