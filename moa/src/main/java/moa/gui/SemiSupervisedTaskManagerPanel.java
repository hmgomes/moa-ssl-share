package moa.gui;

import moa.core.StringUtils;
import moa.options.ClassOption;
import moa.options.OptionHandler;
import moa.tasks.*;
import nz.ac.waikato.cms.gui.core.BaseFileChooser;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.*;
import java.util.ArrayList;
import java.util.prefs.Preferences;

public class SemiSupervisedTaskManagerPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    public static final int MILLISECS_BETWEEN_REFRESH = 600;

    public static String exportFileExtension = "log";

    public class ProgressCellRenderer extends JProgressBar implements
            TableCellRenderer {

        private static final long serialVersionUID = 1L;

        public ProgressCellRenderer() {
            super(SwingConstants.HORIZONTAL, 0, 10000);
            setBorderPainted(false);
            setStringPainted(true);
        }

        @Override
        public Component getTableCellRendererComponent(JTable table,
                                                       Object value, boolean isSelected, boolean hasFocus, int row,
                                                       int column) {
            double frac = -1.0;
            if (value instanceof Double) {
                frac = ((Double) value).doubleValue();
            }
            if (frac >= 0.0) {
                setIndeterminate(false);
                setValue((int) (frac * 10000.0));
                setString(StringUtils.doubleToString(frac * 100.0, 2, 2));
            } else {
                setValue(0);
            }
            return this;
        }

        @Override
        public void validate() { }

        @Override
        public void revalidate() { }

        @Override
        protected void firePropertyChange(String propertyName, Object oldValue,
                                          Object newValue) { }

        @Override
        public void firePropertyChange(String propertyName, boolean oldValue,
                                       boolean newValue) { }
    }

    protected class TaskTableModel extends AbstractTableModel {

        private static final long serialVersionUID = 1L;

        @Override
        public String getColumnName(int col) {
            switch (col) {
                case 0:
                    return "command";
                case 1:
                    return "status";
                case 2:
                    return "time elapsed";
                case 3:
                    return "current activity";
                case 4:
                    return "% complete";
            }
            return null;
        }

        @Override
        public int getColumnCount() {
            return 5;
        }

        @Override
        public int getRowCount() {
            return SemiSupervisedTaskManagerPanel.this.taskList.size();
        }

        @Override
        public Object getValueAt(int row, int col) {
            TaskThread thread = SemiSupervisedTaskManagerPanel.this.taskList.get(row);
            switch (col) {
                case 0:
                    return ((OptionHandler) thread.getTask()).getCLICreationString(SemiSupervisedMainTask.class);
                case 1:
                    return thread.getCurrentStatusString();
                case 2:
                    return StringUtils.secondsToDHMSString(thread.getCPUSecondsElapsed());
                case 3:
                    return thread.getCurrentActivityString();
                case 4:
                    return new Double(thread.getCurrentActivityFracComplete());
            }
            return null;
        }

        @Override
        public boolean isCellEditable(int row, int col) {
            return false;
        }
    }

    protected SemiSupervisedMainTask currentTask;

    protected java.util.List<TaskThread> taskList = new ArrayList<>();

    protected JButton configureTaskButton = new JButton("Configure");

    protected JTextField taskDescField = new JTextField();

    protected JButton runTaskButton = new JButton("Run");

    protected SemiSupervisedTaskManagerPanel.TaskTableModel taskTableModel;

    protected JTable taskTable;

    protected JButton pauseTaskButton = new JButton("Pause");

    protected JButton resumeTaskButton = new JButton("Resume");

    protected JButton cancelTaskButton = new JButton("Cancel");

    protected JButton deleteTaskButton = new JButton("Delete");

    protected PreviewPanel previewPanel;

    private Preferences prefs;

    private final String PREF_NAME = "currentTask";

    public SemiSupervisedTaskManagerPanel() {
        // Read current task preference
        prefs = Preferences.userRoot().node(this.getClass().getName());
        currentTask = new EvaluateInterleavedTestThenTrainSSLDelayed();
        String taskText = this.currentTask.getCLICreationString(SemiSupervisedMainTask.class);
        String propertyValue = prefs.get(PREF_NAME, taskText);
        //this.taskDescField.setText(propertyValue);
        setTaskString(propertyValue, false); //Not store preference
        this.taskDescField.setEditable(false);

        final Component comp = this.taskDescField;
        this.taskDescField.addMouseListener(new MouseAdapter() {

            @Override
            public void mouseClicked(MouseEvent evt) {
                if (evt.getClickCount() == 1) {
                    if ((evt.getButton() == MouseEvent.BUTTON3)
                            || ((evt.getButton() == MouseEvent.BUTTON1) && evt.isAltDown() && evt.isShiftDown())) {
                        JPopupMenu menu = new JPopupMenu();
                        JMenuItem item;

                        item = new JMenuItem("Copy configuration to clipboard");
                        item.addActionListener(new ActionListener() {

                            @Override
                            public void actionPerformed(ActionEvent e) {
                                copyClipBoardConfiguration();
                            }
                        });
                        menu.add(item);

                        item = new JMenuItem("Save selected tasks to file");
                        item.addActionListener(new ActionListener() {

                            @Override
                            public void actionPerformed(ActionEvent arg0) {
                                saveLogSelectedTasks();
                            }
                        });
                        menu.add(item);


                        item = new JMenuItem("Enter configuration...");
                        item.addActionListener(new ActionListener() {

                            @Override
                            public void actionPerformed(ActionEvent arg0) {
                                String newTaskString = JOptionPane.showInputDialog("Insert command line");
                                if (newTaskString != null) {
                                    setTaskString(newTaskString);
                                }
                            }
                        });
                        menu.add(item);

                        menu.show(comp, evt.getX(), evt.getY());
                    }
                }
            }
        });

        JPanel configPanel = new JPanel();
        configPanel.setLayout(new BorderLayout());
        configPanel.add(this.configureTaskButton, BorderLayout.WEST);
        configPanel.add(this.taskDescField, BorderLayout.CENTER);
        configPanel.add(this.runTaskButton, BorderLayout.EAST);
        this.taskTableModel = new SemiSupervisedTaskManagerPanel.TaskTableModel();
        this.taskTable = new JTable(this.taskTableModel);
        DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(SwingConstants.CENTER);
        this.taskTable.getColumnModel().getColumn(1).setCellRenderer(
                centerRenderer);
        this.taskTable.getColumnModel().getColumn(2).setCellRenderer(
                centerRenderer);
        this.taskTable.getColumnModel().getColumn(4).setCellRenderer(
                new SemiSupervisedTaskManagerPanel.ProgressCellRenderer());
        JPanel controlPanel = new JPanel();
        controlPanel.add(this.pauseTaskButton);
        controlPanel.add(this.resumeTaskButton);
        controlPanel.add(this.cancelTaskButton);
        controlPanel.add(this.deleteTaskButton);
        setLayout(new BorderLayout());
        add(configPanel, BorderLayout.NORTH);
        add(new JScrollPane(this.taskTable), BorderLayout.CENTER);
        add(controlPanel, BorderLayout.SOUTH);
        this.taskTable.getSelectionModel().addListSelectionListener(
                new ListSelectionListener() {

                    @Override
                    public void valueChanged(ListSelectionEvent arg0) {
                        taskSelectionChanged();
                    }
                });
        this.configureTaskButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                String newTaskString = ClassOptionSelectionPanel.showSelectClassDialog(
                        SemiSupervisedTaskManagerPanel.this,
                        "Configure task", SemiSupervisedMainTask.class,
                        SemiSupervisedTaskManagerPanel.this.currentTask.getCLICreationString(SemiSupervisedMainTask.class),
                        null);
                setTaskString(newTaskString);
            }
        });
        this.runTaskButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                runTask((Task) SemiSupervisedTaskManagerPanel.this.currentTask.copy());
            }
        });
        this.pauseTaskButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                pauseSelectedTasks();
            }
        });
        this.resumeTaskButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                resumeSelectedTasks();
            }
        });
        this.cancelTaskButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                cancelSelectedTasks();
            }
        });
        this.deleteTaskButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent arg0) {
                deleteSelectedTasks();
            }
        });

        javax.swing.Timer updateListTimer = new javax.swing.Timer(
                MILLISECS_BETWEEN_REFRESH, new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                SemiSupervisedTaskManagerPanel.this.taskTable.repaint();
            }
        });
        updateListTimer.start();
        setPreferredSize(new Dimension(0, 200));
    }

    public void setPreviewPanel(PreviewPanel previewPanel) {
        this.previewPanel = previewPanel;
    }

    public void setTaskString(String cliString) {
        setTaskString(cliString, true);
    }

    public void setTaskString(String cliString, boolean storePreference) {
        try {
            this.currentTask = (SemiSupervisedMainTask) ClassOption.cliStringToObject(
                    cliString, SemiSupervisedMainTask.class, null);
            String taskText = this.currentTask.getCLICreationString(SemiSupervisedMainTask.class);
            this.taskDescField.setText(taskText);
            if (storePreference) {
                //Save task text as a preference
                prefs.put(PREF_NAME, taskText);
            }
        } catch (Exception ex) {
            GUIUtils.showExceptionDialog(this, "Problem with task", ex);
        }
    }

    public void runTask(Task task) {
        TaskThread thread = new TaskThread(task);
        this.taskList.add(0, thread);
        this.taskTableModel.fireTableDataChanged();
        this.taskTable.setRowSelectionInterval(0, 0);
        thread.start();
    }

    public void taskSelectionChanged() {
        TaskThread[] selectedTasks = getSelectedTasks();
        if (selectedTasks.length == 1) {
            setTaskString(((OptionHandler) selectedTasks[0].getTask()).getCLICreationString(SemiSupervisedMainTask.class));
            if (this.previewPanel != null) {
                this.previewPanel.setTaskThreadToPreview(selectedTasks[0]);
            }
        } else {
            this.previewPanel.setTaskThreadToPreview(null);
        }
    }

    public TaskThread[] getSelectedTasks() {
        int[] selectedRows = this.taskTable.getSelectedRows();
        TaskThread[] selectedTasks = new TaskThread[selectedRows.length];
        for (int i = 0; i < selectedRows.length; i++) {
            selectedTasks[i] = this.taskList.get(selectedRows[i]);
        }
        return selectedTasks;
    }

    public void pauseSelectedTasks() {
        TaskThread[] selectedTasks = getSelectedTasks();
        for (TaskThread thread : selectedTasks) {
            thread.pauseTask();
        }
    }

    public void resumeSelectedTasks() {
        TaskThread[] selectedTasks = getSelectedTasks();
        for (TaskThread thread : selectedTasks) {
            thread.resumeTask();
        }
    }

    public void cancelSelectedTasks() {
        TaskThread[] selectedTasks = getSelectedTasks();
        for (TaskThread thread : selectedTasks) {
            thread.cancelTask();
        }
    }

    public void deleteSelectedTasks() {
        TaskThread[] selectedTasks = getSelectedTasks();
        for (TaskThread thread : selectedTasks) {
            thread.cancelTask();
            this.taskList.remove(thread);
        }
        this.taskTableModel.fireTableDataChanged();
    }

    public void copyClipBoardConfiguration() {

        StringSelection selection = new StringSelection(this.taskDescField.getText().trim());
        Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
        clipboard.setContents(selection, selection);

    }

    public void saveLogSelectedTasks() {
        String tasksLog = "";
        TaskThread[] selectedTasks = getSelectedTasks();
        for (TaskThread thread : selectedTasks) {
            tasksLog += ((OptionHandler) thread.getTask()).getCLICreationString(SemiSupervisedMainTask.class) + "\n";
        }

        BaseFileChooser fileChooser = new BaseFileChooser();
        fileChooser.setAcceptAllFileFilterUsed(true);
        fileChooser.addChoosableFileFilter(new FileExtensionFilter(
                exportFileExtension));
        if (fileChooser.showSaveDialog(this) == BaseFileChooser.APPROVE_OPTION) {
            File chosenFile = fileChooser.getSelectedFile();
            String fileName = chosenFile.getPath();
            if (!chosenFile.exists()
                    && !fileName.endsWith(exportFileExtension)) {
                fileName = fileName + "." + exportFileExtension;
            }
            try {
                PrintWriter out = new PrintWriter(new BufferedWriter(
                        new FileWriter(fileName)));
                out.write(tasksLog);
                out.close();
            } catch (IOException ioe) {
                GUIUtils.showExceptionDialog(
                        this,
                        "Problem saving file " + fileName, ioe);
            }
        }
    }

    private static void createAndShowGUI() {

        // Create and set up the labeledInstancesBuffer.
        JFrame frame = new JFrame("Test");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create and set up the content pane.
        JPanel panel = new SemiSupervisedTabPanel();
        panel.setOpaque(true); // content panes must be opaque
        frame.setContentPane(panel);

        // Display the labeledInstancesBuffer.
        frame.pack();
        // frame.setSize(400, 400);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            javax.swing.SwingUtilities.invokeLater(new Runnable() {
                @Override
                public void run() {
                    createAndShowGUI();
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
