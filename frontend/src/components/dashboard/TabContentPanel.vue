<template>
  <section class="tab-content-panel">
    <OverviewContent 
      v-if="activeTabId === 'overview'" 
      :key="`overview-${caseKey}`"
      :case-data="activeCaseData?.overview || activeCaseData" 
    />
    <TimelineContent 
      v-if="activeTabId === 'timeline'" 
      :key="`timeline-${caseKey}`"
      :case-data="activeCaseData" 
    />
    
    <!-- ✅ Pass activeCaseData directly to FinancialContent -->
    <FinancialContent 
      v-if="activeTabId === 'financial'" 
      :key="`financial-${caseKey}`"
      :case-data="activeCaseData" 
    />
    
    <LogisticsContent 
      v-if="activeTabId === 'logistics'" 
      :key="`logistics-${caseKey}`"
      :case-data="activeCaseData?.logisticsData || activeCaseData" 
    />
    <ReportsContent 
      v-if="activeTabId === 'reports'" 
      :key="`reports-${caseKey}`"
      :case-data="activeCaseData?.reportsData || activeCaseData" 
    />
    <DocumentsContent 
      v-if="activeTabId === 'documents'" 
      :key="`documents-${caseKey}`"
      :active-case-data="activeCaseData" 
      :active-case-id="activeCaseData?.id || activeCaseData?.caseNumber || activeCaseId" 
    />

    <div v-if="!isKnownTab" class="tab-content-section placeholder-content">
      <p>Content for '{{ activeTabId }}' is not yet implemented or tab ID is unknown.</p>
    </div>
  </section>
</template>

<script setup>
import { defineProps, computed, watch, ref } from 'vue';

import OverviewContent from './tabs/OverviewContent.vue';
import TimelineContent from './tabs/TimelineContent.vue';
import FinancialContent from './tabs/FinancialContent.vue';
import LogisticsContent from './tabs/LogisticsContent.vue';
import ReportsContent from './tabs/ReportsContent.vue';
import DocumentsContent from './tabs/DocumentsContent.vue'; 

const props = defineProps({
  activeTabId: {
    type: String,
    required: true,
    default: 'overview'
  },
  activeCaseId: { 
    type: String,
    default: null
  },
  activeCaseData: { 
    type: Object,
    default: () => null
  }
});

// ✅ Create a unique key based on case ID to force component re-render
const caseKey = computed(() => {
  const caseId = props.activeCaseData?.caseNumber || props.activeCaseData?.id || props.activeCaseId || 'none';
  return `${caseId}-${Date.now()}`;
});

// ✅ Track case changes for logging
watch(() => props.activeCaseData, (newData, oldData) => {
  const newCaseId = newData?.caseNumber || newData?.id;
  const oldCaseId = oldData?.caseNumber || oldData?.id;
  
  console.log('[TabContentPanel] 🔄 Case data changed');
  console.log('[TabContentPanel]   • Previous Case:', oldCaseId);
  console.log('[TabContentPanel]   • New Case:', newCaseId);
  console.log('[TabContentPanel]   • Current Tab:', props.activeTabId);
  console.log('[TabContentPanel]   • Documents:', newData?.caseDocuments?.length || 0);
  console.log('[TabContentPanel]   • New caseKey:', caseKey.value);
}, { deep: true, immediate: true });

// Watch for tab changes
watch(() => props.activeTabId, (newTabId, oldTabId) => {
  console.log(`[TabContentPanel] Tab changed: '${oldTabId}' → '${newTabId}'`);
}, { immediate: true });

const knownTabIds = ['overview', 'timeline', 'financial', 'logistics', 'reports', 'documents'];
const isKnownTab = computed(() => knownTabIds.includes(props.activeTabId));

</script>

<style scoped>
.tab-content-panel {
  flex: 1; 
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  display: flex; 
  flex-direction: column; 
  overflow-y: auto; 
}

.placeholder-content {
    padding: var(--space-md);
    text-align: center;
    color: #777;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>