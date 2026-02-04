<template>
  <main
    class="main-dashboard-layout"
    :style="{ marginLeft: sidebarMarginLeft }"
  >
    <CaseDetailsPanel
      :active-case-data="activeCaseDetails"
      @tab-selected="handleTabSelection"
    />

    <div class="ai-assistant-area">
      <TabContentPanel
        :active-case-id="activeCaseDetails?.id || activeCaseDetails?.caseNumber"
        :active-tab-id="selectedTabId"
        :active-case-data="activeCaseDetails"
        :style="{ flexGrow: tabFlexGrow, transition: 'flex-grow 0.3s ease' }"
        class="content-panel"
        :class="{ 'expanded': isChatCollapsed }"
      />

      <div
        v-if="!isChatCollapsed"
        class="panel-resizer"
        @click="togglePanelExpansion"
        :title="isChatPanelExpanded ? 'Expand Info Panel' : 'Expand Chat Panel'"
      >
        <i :class="isChatPanelExpanded ? 'fas fa-angle-right' : 'fas fa-angle-left'"></i>
      </div>

      <ChatInterface
        :active-case-id="activeCaseDetails?.id || activeCaseDetails?.caseNumber"
        :style="{ flexGrow: chatFlexGrow, transition: 'flex-grow 0.3s ease' }"
        class="content-panel"
        :class="{ 'panel-collapsed': !isChatPanelExpanded && totalFlexGrow > 2, 'fully-collapsed': isChatCollapsed }"
        @panel-collapsed="handleChatPanelCollapsed"
      />
    </div>
  </main>
</template>

<script setup>
import { defineProps, ref, watch, computed, onMounted } from 'vue';
import CaseDetailsPanel from '../dashboard/CaseDetailsPanel.vue';
import TabContentPanel from '../dashboard/TabContentPanel.vue';
import ChatInterface from '../chat/ChatInterface.vue';
import { useCaseStore } from '@/stores/caseStore';

const props = defineProps({
  isSidebarCollapsed: Boolean
});

const caseStore = useCaseStore();
const selectedTabId = ref('overview');
const isChatPanelExpanded = ref(false);
const isChatCollapsed = ref(false);

const baseTabFlex = 1.5;
const baseChatFlex = 1;
const expandedFlex = 2.5;
const collapsedFlex = 0.5;

const activeCaseDetails = computed(() => {
  return caseStore.activeCaseDetails;
});

const sidebarMarginLeft = computed(() => {
  const expandedSidebarWidth = "250px";
  const collapsedSidebarWidth = "70px";
  return props.isSidebarCollapsed ? collapsedSidebarWidth : expandedSidebarWidth;
});

const tabFlexGrow = computed(() => {
  if (isChatCollapsed.value) {
    return 1;
  }
  return isChatPanelExpanded.value ? collapsedFlex : baseTabFlex;
});

const chatFlexGrow = computed(() => {
  if (isChatCollapsed.value) {
    return 0;
  }
  return isChatPanelExpanded.value ? expandedFlex : baseChatFlex;
});

const totalFlexGrow = computed(() => tabFlexGrow.value + chatFlexGrow.value);

const handleTabSelection = (tabId) => {
  console.log(`[MainDashboardLayout] Received 'tab-selected' event with: ${tabId}`);
  selectedTabId.value = tabId;
  console.log(`[MainDashboardLayout] selectedTabId is now: ${selectedTabId.value}`);
};

const togglePanelExpansion = () => {
  isChatPanelExpanded.value = !isChatPanelExpanded.value;
  console.log(`[MainDashboardLayout] Chat panel expanded: ${isChatPanelExpanded.value}`);
};

const handleChatPanelCollapsed = (collapsed) => {
  isChatCollapsed.value = collapsed;
  console.log(`[MainDashboardLayout] Chat panel fully collapsed: ${collapsed}`);

  if (collapsed) {
    isChatPanelExpanded.value = false;
  }
};

onMounted(() => {
  console.log('[MainDashboardLayout] Mounted with active case:', caseStore.activeCaseDetails?.caseNumber);
  selectedTabId.value = 'overview';
});

</script>

<style scoped>
.main-dashboard-layout {
  flex-grow: 1;
  padding: var(--space-md);
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow-y: hidden;
  transition: margin-left 0.3s ease;
}

.ai-assistant-area {
  display: flex;
  flex-grow: 1;
  overflow: hidden;
  margin-top: var(--space-sm);
  align-items: stretch;
}

.content-panel {
  min-width: 400px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
}

.content-panel.expanded {
  flex-grow: 1 !important;
  min-width: 100%;
}

.content-panel.fully-collapsed {
  flex-grow: 0 !important;
  min-width: 50px !important;
  max-width: 50px !important;
  overflow: visible !important;
}

.panel-resizer {
  flex-shrink: 0;
  width: 12px;
  background-color: var(--light);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border);
  border-right: 1px solid var(--border);
  z-index: 10;
  transition: background-color 0.2s ease;
}

.panel-resizer:hover {
  background-color: #e0e0e0;
}

.panel-resizer i {
  color: var(--secondary);
  font-size: 0.9rem;
}
</style>
