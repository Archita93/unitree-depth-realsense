#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<unitree::robot::go2::publisher::LowCmd>();
    FSMState::lowstate = std::make_shared<unitree::robot::go2::subscription::LowState>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

enum FSMMode
{
    Passive = 1,
    FixStand = 2,
    Velocity = 3,
};

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Initialize FSM
    auto & joy = FSMState::lowstate->joystick;
    auto fsm = std::make_unique<CtrlFSM>(new State_Passive(FSMMode::Passive));
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                // std::cout << joy.LT.pressed << "\n";
                // std::cout << joy.A.pressed << "\n";
                // std::cout << "----\n";
                return joy.LT.pressed && joy.A.pressed;
            }, // L2 + A
            // [&]()->bool{ 
            //     std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            //     int elapsed = std::chrono::duration_cast<std::chrono::seconds> (now - begin).count();
            //     return elapsed > 5;
            // },
            (int)FSMMode::FixStand
        )
    );
    fsm->add(new State_FixStand(FSMMode::FixStand));
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                // std::cout << joy.start.pressed << "\n";
                return joy.start.pressed;
            }, // Start
            // [&]()->bool{ 
            //     std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            //     int elapsed = std::chrono::duration_cast<std::chrono::seconds> (now - begin).count();
            //     return elapsed > 10;
            // },
            FSMMode::Velocity
        )
    );
    fsm->add(new State_RLBase(FSMMode::Velocity, "Velocity"));

    // auto fsm = std::make_unique<CtrlFSM>(new State_RLBase(FSMMode::Velocity, "Velocity"));

    std::cout << "Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "And then press [Start] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

