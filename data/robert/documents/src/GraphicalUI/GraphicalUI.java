package GraphicalUI;

import javax.swing.*;
import java.awt.*;

public class GraphicalUI extends JFrame {

    String [] category = {"Buss", "Tunnelbana               ", "Tåg"};
    JList <String> categories = new JList<> (category);
    private ImagePanel ip = null;
    private Listener listener = new Listener(this);

    public GraphicalUI() {
        super("Register");

//        GraphicalUI.set



        JMenuBar mb = new JMenuBar();
        JMenu iMen = new JMenu ("Archive");
        setJMenuBar(mb);
        mb.add(iMen);
        JMenuItem newMap = new JMenuItem ("New Map");
        iMen.add(newMap);
        newMap.addActionListener(listener.getOpenListener());
        JMenuItem loadPlaces = new JMenuItem ("Load Places");
        iMen.add(loadPlaces);
        JMenuItem save = new JMenuItem ("Save");
        iMen.add(save);
        JMenuItem exit = new JMenuItem ("Exit");
        iMen.add(exit);
        exit.addActionListener(listener.getExitListener());

        setLayout(new BorderLayout());
//        JPanel north = new JPanel();
        JLabel labelTop = new JLabel("Top", SwingConstants.CENTER);
        add(labelTop, BorderLayout.NORTH);

//        JPanel center = new JPanel();
//        center.setLayout(new BorderLayout());
//        add(center, BorderLayout.CENTER);
//        center.setPreferredSize(new Dimension(getWidth(),getHeight()));

        JPanel east = new JPanel();
        east.setLayout(new BorderLayout());
        JLabel labelSort = new JLabel();

        JPanel eastside = new JPanel();
        eastside.setLayout(new BoxLayout(eastside, BoxLayout.Y_AXIS));
        east.add(eastside, BorderLayout.EAST);

        JLabel kategorier = new JLabel("Categories        ");
        eastside.add(kategorier);
        kategorier.setAlignmentX(RIGHT_ALIGNMENT);


        categories.setMinimumSize(new Dimension(150,450));
        categories.setAlignmentX(RIGHT_ALIGNMENT);
        eastside.add(categories);
        categories.addListSelectionListener(listener.getCategoryListener());
//


        JButton hideCategoryButton = new JButton("Hide category");
        hideCategoryButton.setAlignmentX(RIGHT_ALIGNMENT);
        eastside.add(hideCategoryButton);


       eastside.setMinimumSize(new Dimension (150, 450));

        add(eastside, BorderLayout.EAST);

        JPanel north = new JPanel();
        String[] choice = {"", "Named", "Described"};
        JComboBox<String> comboChoice = new JComboBox<String>(choice);
        north.add(new JLabel("New:"));
        north.add(comboChoice);
        comboChoice.addActionListener(listener.getButtonPressed());
        System.out.println("tju");

        JTextField write = new JTextField("Search", 10);
        north.add(write);

        JButton searchButton = new JButton ("Search");
        north.add(searchButton);

        JButton hideButton = new JButton("Hide");
        north.add(hideButton);
        System.out.println("tju");

        JButton removeButton = new JButton ("Remove");
        north.add(removeButton);

        JButton whatButton = new JButton("What is here?");
        north.add(whatButton);
        System.out.println("tju");


        add(north, BorderLayout.NORTH);


        setVisible(true);
//        setSize(900, 500);
        pack();
        setLocationRelativeTo(null);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }
    public String getJList(){
        return categories.getSelectedValue();
    }

    public class Visa extends JFrame {

    }




    public static void main(String[] args) {
        GraphicalUI window = new GraphicalUI();
    }
}

